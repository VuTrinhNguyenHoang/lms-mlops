import { BarChart3, Bell, ChevronDown, HelpCircle, History, UploadCloud } from "lucide-react";
import { NavLink, useLocation } from "react-router-dom";
import type { ReactNode } from "react";

interface LayoutProps {
  children: ReactNode;
}

const navItems = [
  { to: "/", label: "History", icon: History },
  { to: "/batches/new", label: "Upload", icon: UploadCloud },
  { to: "/evaluation", label: "Evaluation", icon: BarChart3 },
];

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const title = getPageTitle(location.pathname);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <span className="brand-bars">
              <i />
              <i />
              <i />
            </span>
          </div>
          <div>
            <div className="brand-title">LMS Risk Portal</div>
          </div>
        </div>

        <nav className="nav-list" aria-label="Main navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <a className="help-link" href="/docs/data-contract.md" target="_blank" rel="noreferrer">
            <HelpCircle size={18} />
            <span>Help</span>
          </a>
          <div className="profile-card">
            <div className="profile-avatar">LA</div>
            <div>
              <div className="profile-name">Learning Analytics</div>
              <div className="profile-email">analytics@example.edu</div>
            </div>
            <ChevronDown size={16} />
          </div>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <h1>{title}</h1>
          <div className="topbar-actions">
            <button className="icon-button" type="button" aria-label="Notifications">
              <Bell size={18} />
            </button>
            <button className="user-menu" type="button">
              <span>LA</span>
              <ChevronDown size={16} />
            </button>
          </div>
        </header>
        <main className="main-surface">{children}</main>
      </div>
    </div>
  );
}

function getPageTitle(pathname: string): string {
  if (pathname === "/") {
    return "Batch History";
  }
  if (pathname === "/batches/new") {
    return "Upload Prediction";
  }
  if (pathname === "/evaluation") {
    return "Evaluation";
  }
  if (pathname.includes("/evaluation")) {
    return "Evaluation Summary";
  }
  if (pathname.startsWith("/batches/")) {
    return "Batch Detail";
  }
  return "LMS Risk Portal";
}
