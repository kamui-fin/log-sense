import { useState } from "react";
import { Group, Code } from "@mantine/core";
import {
    IconBellRinging,
    IconFingerprint,
    IconKey,
    IconSettings,
    Icon2fa,
    IconDatabaseImport,
    IconReceipt2,
    IconSwitchHorizontal,
    IconLogout,
} from "@tabler/icons-react";
import { MantineLogo } from "@mantinex/mantine-logo";
import classes from "./NavbarSimple.module.css";
import Link from "next/link";
import { Router } from "next/router";
import { usePathname } from "next/navigation";

const data = [
    { link: "/", label: "Anomalies", icon: IconBellRinging },
    { link: "/services", label: "Services", icon: IconDatabaseImport },
    {
        link: "/general-settings",
        label: "General Settings",
        icon: IconSettings,
    },
];

export function NavbarSimple() {
    // set active page based on window url
    const [active, setActive] = useState("");
    const pathname = usePathname();
    const links = data.map((item) => (
        <Link
            className={classes.link}
            data-active={
                active === item.link || item.link === pathname
                    ? true
                    : undefined
            }
            href={item.link}
            key={item.label}
            onClick={(event) => {
                setActive(item.label);
            }}
        >
            <item.icon className={classes.linkIcon} stroke={1.5} />
            <span>{item.label}</span>
        </Link>
    ));

    return (
        <nav className={classes.navbar}>
            <div className={classes.navbarMain}>
                <Group className={classes.header} justify="space-between">
                    <MantineLogo size={28} />
                    <Code fw={700}>v3.1.2</Code>
                </Group>
                {links}
            </div>

            {/* <div className={classes.footer}>
        <a
          href="#"
          className={classes.link}
          onClick={(event) => event.preventDefault()}
        >
          <IconSwitchHorizontal className={classes.linkIcon} stroke={1.5} />
          <span>Change account</span>
        </a>

        <a
          href="#"
          className={classes.link}
          onClick={(event) => event.preventDefault()}
        >
          <IconLogout className={classes.linkIcon} stroke={1.5} />
          <span>Logout</span>
        </a>
      </div> */}
        </nav>
    );
}
