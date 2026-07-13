import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  getPlatform: () => ipcRenderer.invoke('get-platform'),
  onScanProgress: (callback: (data: unknown) => void) => {
    ipcRenderer.on('scan-progress', (_event, data) => callback(data));
  },
  onScanComplete: (callback: (data: unknown) => void) => {
    ipcRenderer.on('scan-complete', (_event, data) => callback(data));
  },
});