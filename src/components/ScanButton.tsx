import React from 'react';

const ScanButton: React.FC = () => {
  const [scanning, setScanning] = React.useState(false);

  const handleScan = async () => {
    setScanning(true);
    // TODO: Connect to scanning engine
    // Simulate scan for now
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setScanning(false);
  };

  return (
    <button
      className="scan-button"
      onClick={handleScan}
      disabled={scanning}
    >
      {scanning ? (
        <>⏳ Scanning library...</>
      ) : (
        <>
          <span>🔍</span>
          Scan My Library
        </>
      )}
    </button>
  );
};

export default ScanButton;