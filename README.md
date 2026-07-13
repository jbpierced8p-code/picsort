# PicSort AI

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)]()

AI-powered desktop app that scans photo and video libraries to intelligently find and remove duplicates, group faces with facial recognition, and auto-organize media across devices.

## Features

- **Duplicate Detection** — Finds exact duplicates (SHA-256) and near-duplicates (perceptual hashing)
- **Facial Recognition** — Groups photos by recognized faces (premium tier)
- **Auto-Organize** — Sorts media into folders by date, event, or face group
- **Cross-Device Sync** — Syncs organization rules across devices (premium)
- **Storage Reclamation** — Identifies duplicate files and reclaims wasted space

## Project Structure

```
picsort/
├── app/              # Electron main process
│   ├── main.ts       # Main process entry point
│   └── preload.ts    # Preload script (context bridge)
├── src/              # React renderer (UI)
│   ├── App.tsx       # Root component
│   ├── components/   # UI components
│   └── styles.css    # Application styles
├── engine/           # Python scanning engine
│   ├── scanner.py    # Core scanning & duplicate detection
│   └── requirements.txt
├── ml/               # ML models (face recognition, pHash)
│   └── requirements.txt
├── shared/           # Shared TypeScript types & configs
│   ├── types.ts
│   └── constants.ts
├── scripts/          # Build and utility scripts
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## Getting Started

### Prerequisites

- Node.js >= 18
- Python >= 3.10
- Bun (optional, for faster package installs)

### Installation

```bash
# Install JS dependencies
npm install

# Install Python dependencies
pip install -r engine/requirements.txt
pip install -r ml/requirements.txt
```

### Development

```bash
# Run the React dev server (browser-only)
npm run dev

# Run with Electron
npm run electron:dev
```

### Build

```bash
# Build for production
npm run electron:build
```

## Tiers

| Feature | Free | Premium |
|---------|------|---------|
| Duplicate detection (exact) | ✅ | ✅ |
| Duplicate detection (perceptual) | ✅ | ✅ |
| Scan limit | 5 GB | Unlimited |
| Facial recognition | ❌ | ✅ |
| Auto-cleanup rules | ❌ | ✅ |
| Cross-device sync | ❌ | ✅ |
| Cloud backup integration | ❌ | ✅ |
| Price | Free | $4–$7/mo |

## Tech Stack

- **Desktop Framework:** Electron
- **UI Framework:** React + TypeScript
- **Build Tool:** Vite
- **Backend Engine:** Python
- **ML:** face_recognition, imagehash, OpenCV
- **Packaging:** electron-builder

## License

Private — All rights reserved.