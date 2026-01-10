# Feature F-004: ML Filter Generation (Reinforcement Learning)

---

## Feature ID: F-004

## Feature Name
Machine Learning Filter Generation using Reinforcement Learning

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**High** - Core differentiator of MeshPrep

## Estimated Effort
**XL** (> 1 week)

## Related POC
**POC-07** - RL Pipeline validation

---

## 1. Description

### 1.1 Overview
Use Reinforcement Learning (RL) to automatically discover optimal repair sequences for 3D meshes. The RL agent learns which repair actions to apply based on mesh analysis, with rewards from slicer validation and geometry comparison.

### 1.2 User Story

As a **filter script creator**, I want **the system to suggest or auto-generate repair sequences** so that **I don't need expert knowledge of all possible repair operations**.

### 1.3 Acceptance Criteria

- [ ] RL agent can select repair actions from action space
- [ ] State representation captures mesh issues effectively
- [ ] Reward signal from slicer validation (pass/fail)
- [ ] Reward signal from geometry comparison (Hausdorff)
- [ ] Agent improves repair success rate over time
- [ ] Training runs on GPU (CUDA)
- [ ] Trained model can be saved/loaded
- [ ] User can trigger "Auto-Repair" using trained model

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| MeshModel | MeshModel | Yes | Current mesh state |
| AnalysisResult | MeshAnalysisResult | Yes | Issues detected |
| OriginalMesh | MeshModel | Yes | For geometry comparison |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| RepairAction | RepairAction | Next action to take |
| ActionSequence | List<RepairAction> | Complete repair sequence (episode) |
| Confidence | double | Agent's confidence in the action |

### 2.3 RL Framework

**State Space:**
- Mesh statistics (vertex/face count, volume, surface area)
- Issue counts (non-manifold, holes, intersections, etc.)
- Normalized values for neural network input

**Action Space:**
- Fill holes (various strategies)
- Remove self-intersections
- Fix non-manifold edges
- Fix non-manifold vertices
- Remove degenerate triangles
- Remesh region
- Smooth surface
- Decimate (reduce triangles)
- No-op (done)

**Reward Function:**
```
R = w1 * SlicerPass + w2 * (1 - HausdorffNorm) + w3 * (1 - MeanHausdorffNorm) + w4 * IssueReduction

Where:
- SlicerPass: +1.0 if slicer accepts, -1.0 if fails
- HausdorffNorm: Max deviation normalized (0-1, lower is better)
- MeanHausdorffNorm: Mean deviation normalized (0-1, lower is better)
- IssueReduction: Fraction of issues fixed this step
```

### 2.4 Business Rules

- Max episode length: 20 actions (prevent infinite loops)
- Geometry must not deviate beyond threshold (Hausdorff)
- Final mesh must pass slicer validation for positive reward
- User feedback can boost/penalize specific outcomes

---

## 3. Technical Details

### 3.1 Dependencies

- **TorchSharp** (NuGet) - PyTorch for .NET, GPU/CUDA support
- **MeshLib** - Mesh repair operations
- Slicer CLI - For validation reward

### 3.2 Affected Components

- `MeshPrep.Core` - RL agent, training loop
- `MeshPrep.FilterScriptCreator` - Auto-repair UI, training controls
- SQLite database - Store training experiences

### 3.3 Technical Approach

```
┌─────────────────────────────────────────────────────────┐
│                    RL Training Loop                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │  State  │───►│  Agent  │───►│ Action  │             │
│  │ (Mesh)  │    │  (DQN)  │    │         │             │
│  └─────────┘    └─────────┘    └────┬────┘             │
│       ▲                             │                   │
│       │         ┌───────────────────┘                   │
│       │         ▼                                       │
│       │    ┌─────────┐    ┌─────────┐                  │
│       │    │ MeshLib │───►│  New    │                  │
│       │    │ Repair  │    │  Mesh   │                  │
│       │    └─────────┘    └────┬────┘                  │
│       │                        │                        │
│       │         ┌──────────────┴──────────────┐        │
│       │         ▼                             ▼        │
│       │    ┌─────────┐                  ┌─────────┐   │
│       │    │ Slicer  │                  │Hausdorff│   │
│       │    │  Check  │                  │  Check  │   │
│       │    └────┬────┘                  └────┬────┘   │
│       │         │                            │         │
│       │         └──────────────┬─────────────┘        │
│       │                        ▼                       │
│       │                   ┌─────────┐                  │
│       └───────────────────│ Reward  │                  │
│                           └─────────┘                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.4 API/Interface

```csharp
namespace MeshPrep.Core.ML
{
    public interface IRLAgent
    {
        RepairAction SelectAction(RLState state, bool explore = false);
        void Train(RLExperience experience);
        void SaveModel(string path);
        void LoadModel(string path);
    }

    public class RLState
    {
        public double[] MeshFeatures { get; set; }  // Normalized mesh stats
        public double[] IssueFeatures { get; set; } // Normalized issue counts
    }

    public class RLExperience
    {
        public RLState State { get; set; }
        public RepairAction Action { get; set; }
        public double Reward { get; set; }
        public RLState NextState { get; set; }
        public bool Done { get; set; }
    }

    public enum RepairAction
    {
        FillHoles,
        RemoveSelfIntersections,
        FixNonManifoldEdges,
        FixNonManifoldVertices,
        RemoveDegenerateTriangles,
        RemeshRegion,
        SmoothSurface,
        Decimate,
        Done
    }
}
```

---

## 4. User Interface

### 4.1 UI Changes Required

**FilterScriptCreator:**
- "Auto-Repair" button - Uses trained model to suggest/apply repairs
- Training panel (advanced) - Start/stop training, view progress
- Confidence indicator for suggested actions

### 4.2 User Interaction Flow

```
User clicks "Auto-Repair"
         │
         ▼
    Analyze mesh ──► Generate state ──► Agent selects action
         │                                    │
         │              ┌─────────────────────┘
         │              ▼
         │         Apply action ──► Re-analyze ──► Check done?
         │              ▲                              │
         │              │                    ┌────────┴────────┐
         │              │                    ▼                 ▼
         │              │                  No               Yes
         │              │                    │                 │
         │              └────────────────────┘                 ▼
         │                                              Validate result
         │                                                     │
         │                                    ┌────────────────┴────────────────┐
         │                                    ▼                                 ▼
         │                               Success ✅                         Failed ❌
         │                                    │                                 │
         ▼                                    ▼                                 ▼
    Show repair sequence              Show repaired mesh               Show what failed
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Agent selects valid action | Mesh with holes | FillHoles action | ⬜ |
| TC-002 | Agent learns from reward | Training data | Improved accuracy | ⬜ |
| TC-003 | Save/load model | Trained model | Model persists | ⬜ |
| TC-004 | GPU acceleration | Large batch | GPU utilized | ⬜ |
| TC-005 | Episode terminates | Max 20 steps | Episode ends | ⬜ |
| TC-006 | Reward calculation | Various outcomes | Correct rewards | ⬜ |
| TC-007 | Auto-repair workflow | Broken mesh | Repaired mesh | ⬜ |

### 5.2 Edge Cases

- Mesh that cannot be repaired
- Multiple equally valid actions
- Agent stuck in loop
- GPU not available (fallback to CPU)

---

## 6. Notes & Open Questions

### Open Questions
- [x] Which RL framework? → **TorchSharp (GPU/CUDA support)**
- [x] Which algorithm? → **DQN or PPO (start with DQN)**
- [ ] How to handle continuous action parameters? → **Discretize or use actor-critic**

### Notes
- Pre-train on Thingi10K dataset for good initial policy
- Online learning from user feedback
- Consider experience replay for sample efficiency
- Model should be small enough for fast inference

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
