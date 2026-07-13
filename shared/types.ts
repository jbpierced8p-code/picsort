// Shared types for PicSort AI

export interface ScanResult {
  totalFiles: number;
  duplicates: DuplicateGroup[];
  totalDuplicates: number;
  storageReclaimable: number; // bytes
  facesDetected: number;
  scanDuration: number; // ms
}

export interface DuplicateGroup {
  id: string;
  files: DuplicateFile[];
  algorithm: 'exact' | 'perceptual' | 'metadata';
  confidence: number;
}

export interface DuplicateFile {
  path: string;
  size: number;
  hash: string;
  created: string;
  modified: string;
}

export interface FaceGroup {
  id: string;
  label: string;
  faceCount: number;
  thumbnailPaths: string[];
}

export interface ScanProgress {
  phase: 'scanning' | 'hashing' | 'comparing' | 'done';
  current: number;
  total: number;
  percentage: number;
}

export type AppTier = 'free' | 'premium';

export interface TierLimits {
  maxScanSize: number; // bytes, 0 = unlimited
  facialRecognition: boolean;
  autoCleanup: boolean;
  crossDeviceSync: boolean;
  cloudBackup: boolean;
}

export const TIER_CONFIG: Record<AppTier, TierLimits> = {
  free: {
    maxScanSize: 5 * 1024 * 1024 * 1024, // 5 GB
    facialRecognition: false,
    autoCleanup: false,
    crossDeviceSync: false,
    cloudBackup: false,
  },
  premium: {
    maxScanSize: 0, // unlimited
    facialRecognition: true,
    autoCleanup: true,
    crossDeviceSync: true,
    cloudBackup: true,
  },
};