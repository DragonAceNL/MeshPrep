# POC-07: RL Pipeline

---

## POC ID: POC-07

## POC Name
Reinforcement Learning Pipeline with TorchSharp

## Status
- [x] Not Started
- [ ] In Progress
- [ ] Completed - Success
- [ ] Completed - Failed
- [ ] Blocked

## Estimated Effort
**5-7 days**

## Related Features
- F-004: ML Filter Generation
- F-012: User Feedback System

---

## 1. Objective

### 1.1 What We're Proving
Validate that TorchSharp can implement a reinforcement learning agent that learns optimal mesh repair sequences, with GPU acceleration for training.

### 1.2 Success Criteria

- [ ] TorchSharp runs with CUDA/GPU support
- [ ] DQN agent can be trained on simple task
- [ ] State representation captures mesh issues
- [ ] Actions map to repair operations
- [ ] Reward signal from slicer validation works
- [ ] Reward signal from Hausdorff distance works
- [ ] Agent improves over training episodes
- [ ] Model can be saved and loaded
- [ ] Inference is fast enough for interactive use (<1s)

### 1.3 Failure Criteria

- TorchSharp GPU support not working
- Training unstable or doesn't converge
- RL approach too slow for practical use
- Cannot integrate with mesh repair workflow

---

## 2. Technical Approach

### 2.1 Technologies to Evaluate

| Technology | Version | Purpose |
|------------|---------|---------|
| TorchSharp | Latest NuGet | PyTorch for .NET |
| TorchSharp.Cuda | Latest | GPU acceleration |
| .NET 10 | | Runtime |

### 2.2 RL Framework Design

**State Space:**
```
State = [
    vertex_count_normalized,      # 0-1
    face_count_normalized,        # 0-1
    volume_normalized,            # 0-1
    non_manifold_edge_ratio,      # 0-1
    non_manifold_vertex_ratio,    # 0-1
    hole_count_normalized,        # 0-1
    self_intersection_ratio,      # 0-1
    degenerate_face_ratio,        # 0-1
    is_watertight,                # 0 or 1
    previous_action_1,            # one-hot
    previous_action_2,            # one-hot
    ...
]
```

**Action Space:**
```
0: FillHoles
1: RemoveSelfIntersections
2: FixNonManifoldEdges
3: FixNonManifoldVertices
4: RemoveDegenerateFaces
5: Smooth
6: Decimate
7: Done (terminate episode)
```

**Reward Function:**
```
R = w1 * slicer_reward + w2 * geometry_reward + w3 * issue_reduction

Where:
- slicer_reward: +1.0 if slicer accepts, -1.0 if fails
- geometry_reward: Based on Hausdorff (0.5 to -0.5)
- issue_reduction: Fraction of issues fixed this step
```

### 2.3 Test Scenarios

1. **GPU Setup** - Verify TorchSharp CUDA works
2. **Simple DQN** - Train on CartPole (standard RL test)
3. **State Encoding** - Convert mesh to state vector
4. **Action Execution** - Map action to MeshLib operation
5. **Reward Calculation** - Combine slicer + Hausdorff
6. **Training Loop** - Full training on simple meshes
7. **Convergence** - Agent learns to fix basic issues

### 2.4 Test Data

- Simple meshes with single issue type
- Meshes with 2-3 combined issues
- Thingi10K sample for variety

---

## 3. Implementation

### 3.1 Setup Steps

1. Create new .NET 10 console project: `MeshPrep.POC.RLPipeline`
2. Install NuGet packages:
   ```
   dotnet add package TorchSharp
   dotnet add package TorchSharp-cuda-windows  # or linux
   dotnet add package MeshLib
   ```
3. Verify CUDA setup with simple tensor operations
4. Implement DQN agent
5. Implement mesh environment
6. Train and evaluate

### 3.2 Code Location

`/poc/POC_07_RLPipeline/`

### 3.3 Key Code Snippets

**GPU Verification:**
```csharp
using TorchSharp;
using static TorchSharp.torch;

public void VerifyGpu()
{
    Console.WriteLine($"CUDA available: {cuda.is_available()}");
    
    if (cuda.is_available())
    {
        Console.WriteLine($"CUDA device count: {cuda.device_count()}");
        Console.WriteLine($"CUDA device: {cuda.get_device_name(0)}");
        
        // Test tensor on GPU
        var tensor = torch.randn(1000, 1000, device: CUDA);
        var result = tensor.mm(tensor);
        Console.WriteLine($"GPU computation successful");
    }
}
```

**DQN Network:**
```csharp
public class DQNetwork : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> layers;
    
    public DQNetwork(int stateSize, int actionSize) : base("DQN")
    {
        layers = nn.Sequential(
            nn.Linear(stateSize, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actionSize)
        );
        
        RegisterComponents();
    }
    
    public override Tensor forward(Tensor x)
    {
        return layers.forward(x);
    }
}
```

**Replay Buffer:**
```csharp
public class ReplayBuffer
{
    private readonly List<Experience> buffer = new();
    private readonly int maxSize;
    private readonly Random random = new();
    
    public void Add(Experience exp)
    {
        if (buffer.Count >= maxSize)
            buffer.RemoveAt(0);
        buffer.Add(exp);
    }
    
    public List<Experience> Sample(int batchSize)
    {
        return buffer.OrderBy(_ => random.Next()).Take(batchSize).ToList();
    }
}

public record Experience(
    float[] State,
    int Action,
    float Reward,
    float[] NextState,
    bool Done
);
```

**DQN Agent:**
```csharp
public class DQNAgent
{
    private readonly DQNetwork policyNet;
    private readonly DQNetwork targetNet;
    private readonly optim.Optimizer optimizer;
    private readonly ReplayBuffer buffer;
    private double epsilon = 1.0;
    
    public int SelectAction(float[] state, bool training = true)
    {
        if (training && Random.Shared.NextDouble() < epsilon)
        {
            // Explore
            return Random.Shared.Next(actionSize);
        }
        
        // Exploit
        using var _ = torch.no_grad();
        var stateTensor = torch.tensor(state, device: device).unsqueeze(0);
        var qValues = policyNet.forward(stateTensor);
        return (int)qValues.argmax(1).item<long>();
    }
    
    public void Train(int batchSize)
    {
        if (buffer.Count < batchSize) return;
        
        var batch = buffer.Sample(batchSize);
        
        // Convert to tensors
        var states = torch.tensor(batch.Select(e => e.State).ToArray(), device: device);
        var actions = torch.tensor(batch.Select(e => e.Action).ToArray(), device: device);
        var rewards = torch.tensor(batch.Select(e => e.Reward).ToArray(), device: device);
        var nextStates = torch.tensor(batch.Select(e => e.NextState).ToArray(), device: device);
        var dones = torch.tensor(batch.Select(e => e.Done ? 1f : 0f).ToArray(), device: device);
        
        // Compute Q values
        var qValues = policyNet.forward(states).gather(1, actions.unsqueeze(1));
        
        // Compute target Q values
        using (torch.no_grad())
        {
            var nextQValues = targetNet.forward(nextStates).max(1).values;
            var targetQ = rewards + gamma * nextQValues * (1 - dones);
        }
        
        // Loss and backprop
        var loss = nn.functional.mse_loss(qValues.squeeze(), targetQ);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        // Decay epsilon
        epsilon = Math.Max(0.01, epsilon * 0.995);
    }
}
```

**Mesh Environment:**
```csharp
public class MeshRepairEnvironment
{
    private Mesh originalMesh;
    private Mesh currentMesh;
    private int stepCount;
    private const int MaxSteps = 20;
    
    public float[] Reset(string meshPath)
    {
        originalMesh = Mesh.FromAnySupportedFormat(meshPath);
        currentMesh = originalMesh.Clone();
        stepCount = 0;
        return GetState();
    }
    
    public (float[] nextState, float reward, bool done) Step(int action)
    {
        stepCount++;
        
        // Execute action
        ExecuteAction(action);
        
        // Calculate reward
        var reward = CalculateReward();
        
        // Check if done
        var done = stepCount >= MaxSteps || action == 7 || IsMeshValid();
        
        return (GetState(), reward, done);
    }
    
    private float[] GetState()
    {
        var analysis = AnalyzeMesh(currentMesh);
        return NormalizeState(analysis);
    }
    
    private float CalculateReward()
    {
        float reward = 0;
        
        // Slicer validation
        var slicerValid = ValidateWithSlicer(currentMesh);
        reward += slicerValid ? 1.0f : -0.5f;
        
        // Geometry fidelity
        var hausdorff = CompareGeometry(originalMesh, currentMesh);
        if (hausdorff.MaxHausdorff < 0.5 && hausdorff.MeanHausdorff < 0.05)
        {
            reward += 0.5f * (1 - hausdorff.MeanHausdorff / 0.05f);
        }
        else
        {
            reward -= 0.5f;
        }
        
        return reward;
    }
}
```

**Training Loop:**
```csharp
public void Train(int episodes = 1000)
{
    var env = new MeshRepairEnvironment();
    var agent = new DQNAgent(stateSize: 20, actionSize: 8);
    
    var rewardHistory = new List<float>();
    
    for (int ep = 0; ep < episodes; ep++)
    {
        var state = env.Reset(GetRandomMeshPath());
        float totalReward = 0;
        
        while (true)
        {
            var action = agent.SelectAction(state);
            var (nextState, reward, done) = env.Step(action);
            
            agent.Buffer.Add(new Experience(state, action, reward, nextState, done));
            agent.Train(batchSize: 32);
            
            state = nextState;
            totalReward += reward;
            
            if (done) break;
        }
        
        rewardHistory.Add(totalReward);
        
        if (ep % 100 == 0)
        {
            var avgReward = rewardHistory.TakeLast(100).Average();
            Console.WriteLine($"Episode {ep}, Avg Reward: {avgReward:F2}, Epsilon: {agent.Epsilon:F3}");
        }
    }
    
    agent.Save("model.pt");
}
```

---

## 4. Results

### 4.1 Test Results

| Test | Result | Notes |
|------|--------|-------|
| TorchSharp CUDA | ⬜ | |
| GPU tensor operations | ⬜ | |
| DQN on CartPole | ⬜ | |
| State encoding | ⬜ | |
| Action execution | ⬜ | |
| Slicer reward | ⬜ | |
| Hausdorff reward | ⬜ | |
| Training convergence | ⬜ | |
| Model save/load | ⬜ | |

### 4.2 Performance Metrics

| Metric | Target | Actual | Pass? |
|--------|--------|--------|-------|
| Training episode time | < 30s | | ⬜ |
| GPU utilization | > 50% | | ⬜ |
| Inference time | < 1s | | ⬜ |
| Learning improvement | > 50% | | ⬜ |
| Final success rate | > 70% | | ⬜ |

### 4.3 Training Progress

| Episodes | Avg Reward | Success Rate |
|----------|------------|--------------|
| 100 | | |
| 500 | | |
| 1000 | | |

### 4.4 Issues Encountered

*To be filled during POC execution*

---

## 5. Conclusions

### 5.1 Recommendation
*To be filled after POC completion*

### 5.2 RL Configuration Recommendations
*To be filled after POC completion*

| Parameter | Recommended Value |
|-----------|-------------------|
| Learning rate | |
| Batch size | |
| Epsilon decay | |
| Network architecture | |
| Reward weights | |

### 5.3 Risks Identified
*To be filled after POC completion*

### 5.4 Next Steps
*To be filled after POC completion*

---

## 6. Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | POC document created | |
