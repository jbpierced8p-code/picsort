import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="header-logo">
        <div className="logo-icon">✦</div>
        <span>PicSort AI</span>
      </div>
      <nav>
        <ul className="header-nav">
          <li><a href="#">Scan</a></li>
          <li><a href="#">Albums</a></li>
          <li><a href="#">Settings</a></li>
        </ul>
      </nav>
    </header>
  );
};

export default Header;