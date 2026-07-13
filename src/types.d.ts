// Type declarations for the Electron preload API

interface ElectronAPI {
  getAppVersion: () => Promise<string>;
  getPlatform: () => Promise<string>;
  onScanProgress: (callback: (data: unknown) => void) => void;
  onScanComplete: (callback: (data: unknown) => void) => void;
}

interface Window {
  electronAPI?: ElectronAPI;
}