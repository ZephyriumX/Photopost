# 🤝 Contributing to PhotoPost

## 🔀 Branching Strategy

- All development work happens in the `features` branch.
- Each task/issue gets its own folder inside `/features`.
- Workflow:
  1. Pull latest `features` branch.
  2. Work in the assigned folder for your task/issue.
  3. Commit changes with a message referencing the issue number.
  4. Push to `features` branch.
  5. Once tested, merge `features` → `dev` → `main`.

## 🔧 Commit Messages
Follow this format:
- feat: Add new feature
- fix: Fix a bug
- chore: Non-functional change
- docs: README or documentation changes
  
Example:
```feat: add CLIP-based pose recommender (#1)```

## ✅ Pull Requests
- Always link to the issue (`Closes #X`)
- Fill out PR template
- Wait for code review before merging
