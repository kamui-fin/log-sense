import { NavbarSimple } from "./components/nav/NavbarSimple";

interface LayoutProps {
    children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
    return (
        <main className="flex h-screen">
            <NavbarSimple />
            <div className="pl-8 pr-8 w-full">{children}</div>
        </main>
    );
};
