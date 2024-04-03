import { NavbarSimple } from "./NavbarSimple";

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <main className="flex h-screen">
      <NavbarSimple />
      <div className="pl-8">{children}</div>
    </main>
  );
};
