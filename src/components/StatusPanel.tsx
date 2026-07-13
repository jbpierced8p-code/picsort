import React from 'react';

const StatusPanel: React.FC = () => {
  return (
    <div className="status-panel">
      <h3>Library Status</h3>
      <div className="status-item">
        <span className="status-label">Photos scanned</span>
        <span className="status-value">0</span>
      </div>
      <div className="status-item">
        <span className="status-label">Duplicates found</span>
        <span className="status-value">0</span>
      </div>
      <div className="status-item">
        <span className="status-label">Storage reclaimed</span>
        <span className="status-value">0 GB</span>
      </div>
      <div className="status-item">
        <span className="status-label">Faces detected</span>
        <span className="status-value">0</span>
      </div>
    </div>
  );
};

export default StatusPanel;