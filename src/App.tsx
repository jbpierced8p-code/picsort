import React from 'react';
import Header from './components/Header';
import ScanButton from './components/ScanButton';
import StatusPanel from './components/StatusPanel';

const App: React.FC = () => {
  const [appVersion, setAppVersion] = React.useState<string>('');

  React.useEffect(() => {
    // Check if running in Electron
    if (window.electronAPI) {
      window.electronAPI.getAppVersion().then((version: string) => {
        setAppVersion(version);
      });
    }
  }, []);

  return (
    <div className="app">
      <Header />
      <main className="main-content">
        <div className="hero">
          <h1 className="hero-title">Welcome to PicSort AI</h1>
          <p className="hero-subtitle">
            Intelligently clean and organize your photo library.
            Find duplicates, group faces, and reclaim storage — automatically.
          </p>
          {appVersion && (
            <p className="version-badge">v{appVersion}</p>
          )}
        </div>
        <div className="actions">
          <ScanButton />
          <StatusPanel />
        </div>
      </main>
    </div>
  );
};

export default App;