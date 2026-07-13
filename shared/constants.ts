// App-wide constants

export const APP_NAME = 'PicSort AI';
export const APP_VERSION = '0.1.0';
export const APP_ID = 'ai.picsort.desktop';

export const SUPPORTED_IMAGE_EXTENSIONS = [
  '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic', '.heif',
];

export const SUPPORTED_VIDEO_EXTENSIONS = [
  '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm',
];

export const HASH_ALGORITHM = 'sha256';
export const PERCEPTUAL_HASH_SIZE = 8; // 8x8 for pHash

export const DEFAULT_SCAN_PATHS: string[] = [
  '~/Pictures',
  '~/Desktop',
  '~/Downloads',
];