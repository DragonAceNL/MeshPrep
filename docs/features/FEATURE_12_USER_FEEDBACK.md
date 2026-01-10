# Feature F-012: User Feedback System

---

## Feature ID: F-012

## Feature Name
User Feedback System for RL Training

## Status
- [x] Not Started
- [ ] In Analysis
- [ ] In Development
- [ ] In Testing
- [ ] Complete

## Priority
**Medium** - Improves RL over time

## Estimated Effort
**Medium** (1-3 days)

## Related POC
None - Builds on F-004 (RL)

---

## 1. Description

### 1.1 Overview
Allow users to provide feedback on repair results (good/bad), which feeds into the RL system as additional reward signals. This enables continuous improvement of the auto-repair model.

### 1.2 User Story

As a **FilterScriptCreator user**, I want **to tell the system if a repair was good or bad** so that **it learns from my preferences and improves over time**.

### 1.3 Acceptance Criteria

- [ ] Simple thumbs up/down feedback UI
- [ ] Optional detailed feedback (what was wrong)
- [ ] Store feedback in local database
- [ ] Feed feedback into RL reward signal
- [ ] Show feedback statistics
- [ ] Export feedback data (for model improvement)

---

## 2. Functional Details

### 2.1 Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| RepairSession | RepairSession | Yes | The repair that was performed |
| Rating | Rating | Yes | Good/Bad/Neutral |
| Details | string | No | Optional explanation |
| Issues | List<FeedbackIssue> | No | Specific problems |

### 2.2 Outputs

| Output | Type | Description |
|--------|------|-------------|
| FeedbackRecord | FeedbackRecord | Stored feedback entry |

### 2.3 Feedback Types

| Rating | Description | RL Effect |
|--------|-------------|-----------|
| 👍 Good | Repair worked well | +0.5 bonus reward |
| 👎 Bad | Repair had problems | -0.5 penalty |
| 😐 Neutral | No strong opinion | No adjustment |

### 2.4 Feedback Categories (Optional)

- Geometry changed too much
- Repair incomplete
- New issues introduced
- Took too long
- Perfect result
- Other (free text)

---

## 3. Technical Details

### 3.1 Dependencies

- SQLite database for storage
- F-004 (RL) for reward integration

### 3.2 Affected Components

- `MeshPrep.Core` - Feedback storage and processing
- `MeshPrep.FilterScriptCreator` - Feedback UI

### 3.3 Database Schema

```sql
CREATE TABLE Feedback (
    Id INTEGER PRIMARY KEY,
    SessionId TEXT NOT NULL,
    Timestamp DATETIME NOT NULL,
    Rating INTEGER NOT NULL,  -- -1=Bad, 0=Neutral, 1=Good
    Details TEXT,
    OriginalFingerprint TEXT,
    RepairOperations TEXT,    -- JSON array
    HausdorffMax REAL,
    HausdorffMean REAL,
    SlicerValid BOOLEAN
);

CREATE TABLE FeedbackIssues (
    Id INTEGER PRIMARY KEY,
    FeedbackId INTEGER REFERENCES Feedback(Id),
    IssueType TEXT,
    Description TEXT
);
```

### 3.4 API/Interface

```csharp
namespace MeshPrep.Core.Feedback
{
    public interface IFeedbackService
    {
        void SubmitFeedback(FeedbackRecord feedback);
        List<FeedbackRecord> GetRecentFeedback(int count = 100);
        FeedbackStatistics GetStatistics();
        void ExportFeedback(string filePath);
    }

    public class FeedbackRecord
    {
        public string SessionId { get; set; }
        public DateTime Timestamp { get; set; }
        public Rating Rating { get; set; }
        public string Details { get; set; }
        public List<FeedbackIssue> Issues { get; set; }
        
        // Context
        public string OriginalFingerprint { get; set; }
        public List<string> RepairOperations { get; set; }
        public double HausdorffMax { get; set; }
        public double HausdorffMean { get; set; }
        public bool SlicerValid { get; set; }
    }

    public enum Rating
    {
        Bad = -1,
        Neutral = 0,
        Good = 1
    }

    public class FeedbackStatistics
    {
        public int TotalFeedback { get; set; }
        public int GoodCount { get; set; }
        public int BadCount { get; set; }
        public double AverageRating { get; set; }
        public Dictionary<string, int> IssueFrequency { get; set; }
    }
}
```

---

## 4. User Interface

### 4.1 UI Components

**After Repair Completion:**
```
┌─────────────────────────────────────────┐
│         How was this repair?            │
│                                         │
│    [ 👎 Bad ]  [ 😐 OK ]  [ 👍 Good ]   │
│                                         │
│  (Optional) What could be improved?     │
│  ┌─────────────────────────────────────┐│
│  │                                     ││
│  └─────────────────────────────────────┘│
│                                         │
│  □ Geometry changed too much            │
│  □ Repair incomplete                    │
│  □ New issues introduced                │
│                                         │
│            [ Submit ]  [ Skip ]         │
└─────────────────────────────────────────┘
```

**Feedback Statistics Panel:**
```
Repair Statistics (Last 30 days)
├── Total repairs: 47
├── Good: 38 (81%)
├── Bad: 5 (11%)
└── Neutral: 4 (8%)

Common Issues:
├── Geometry changed: 3
└── Incomplete repair: 2
```

### 4.2 User Interaction Flow

```
Repair Complete ──► Show feedback prompt ──► User rates
                                                │
                              ┌─────────────────┼─────────────────┐
                              ▼                 ▼                 ▼
                           Skip              Quick rate        Detailed
                              │                 │                 │
                              │                 ▼                 ▼
                              │         Store rating      Show form
                              │                                   │
                              │                                   ▼
                              └────────────────┬──────────────────┘
                                               ▼
                                        Store in DB
                                               │
                                               ▼
                                      Update RL reward
```

---

## 5. Testing

### 5.1 Test Cases

| Test ID | Description | Input | Expected Output | Status |
|---------|-------------|-------|-----------------|--------|
| TC-001 | Submit good feedback | Click 👍 | Record stored | ⬜ |
| TC-002 | Submit bad feedback | Click 👎 | Record stored | ⬜ |
| TC-003 | Skip feedback | Click Skip | No record | ⬜ |
| TC-004 | Detailed feedback | Add comments | Stored with details | ⬜ |
| TC-005 | View statistics | Open stats panel | Correct counts | ⬜ |
| TC-006 | Export feedback | Export button | JSON file created | ⬜ |
| TC-007 | RL integration | Good feedback | Reward adjusted | ⬜ |

### 5.2 Edge Cases

- Multiple feedback for same session
- Feedback after app restart
- Database corruption recovery

---

## 6. Notes & Open Questions

### Open Questions
- [x] Require feedback? → **No, optional but encouraged**
- [ ] Share feedback anonymously? → **Consider for future cloud service**

### Notes
- Feedback is local only in v1.0
- Consider gamification (streak badges) to encourage feedback
- Export format allows future centralized learning
- Non-intrusive prompt - easy to skip

---

## 7. Implementation Log

| Date | Update | Author |
|------|--------|--------|
| 2026-01-10 | Feature document created | |
