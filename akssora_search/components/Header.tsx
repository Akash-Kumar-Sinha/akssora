"use client";

import {
  Navbar,
  NavbarLogo,
  NavbarButton,
} from "@/components/ui/resizable-navbar";
import { AnimatedThemeToggler } from "./ui/animated-theme-toggler";
import useUser from "@/lib/hook/useUser";
import { Logout } from "./Logout";
import { BACKEND_URL } from "@/lib/constant";

export const Header = () => {
  const { user } = useUser();

  const handleLogin = () => {
    const auth = `${BACKEND_URL}/auth/oauth/google/login`;
    window.location.href = auth;
  };

  return (
    <>
      <Navbar>
        <div className="flex w-full justify-between p-4 px-6">
          <NavbarLogo />
          <div className="flex items-center gap-2">
            <NavbarButton variant="secondary" className="w-full">
              <AnimatedThemeToggler />
            </NavbarButton>{" "}
            {!user ? (
              <NavbarButton
                variant="primary"
                className="border-2 border-primary/20"
                onClick={handleLogin}
              >
                Login
              </NavbarButton>
            ) : (
              <div className="flex flex-col items-center gap-1">
                {user.Avatar ? (
                  <img
                    src={user.Avatar}
                    alt="Profile"
                    className="w-8 rounded-full"
                  />
                ) : (
                  <p>{user.Username}</p>
                )}
                <Logout />
              </div>
            )}
          </div>
        </div>
      </Navbar>
    </>
  );
};
