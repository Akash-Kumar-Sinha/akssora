"use client";
import { BACKEND_URL } from "@/lib/constant";
import useUser from "@/lib/hook/useUser";
import { AnimatePresence, motion } from "motion/react";
import React, { useEffect, useRef, useState } from "react";
import { Logout } from "./Logout";

export const Login = () => {
  const { user } = useUser();
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleLogin = () => {
    window.location.href = `${BACKEND_URL}/auth/oauth/google/login`;
  };

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  if (!user) {
    return (
      <motion.button
        whileHover={{ scale: 1.04 }}
        whileTap={{ scale: 0.96 }}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
        onClick={handleLogin}
        className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-white text-black text-sm font-semibold whitespace-nowrap shadow-md hover:shadow-white/20 transition-shadow"
      >
        Sign in
      </motion.button>
    );
  }

  return (
    <div ref={containerRef} className="relative">
      <motion.button
        whileHover={{ scale: 1.06 }}
        whileTap={{ scale: 0.94 }}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-full focus:outline-none"
        aria-label="Account menu"
      >
        {user.Avatar ? (
          <img
            src={user.Avatar}
            alt={user.Username ?? "Profile"}
            referrerPolicy="no-referrer"
            className="w-9 h-9 rounded-full object-cover ring-2 ring-white/20 hover:ring-white/50 transition-all duration-200"
          />
        ) : (
          <div className="w-9 h-9 rounded-full bg-white/10 ring-2 ring-white/20 hover:ring-white/50 transition-all duration-200 flex items-center justify-center">
            <span className="text-white text-sm font-bold uppercase">
              {user.Username?.charAt(0) ?? "U"}
            </span>
          </div>
        )}
      </motion.button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 400, damping: 28 }}
            className="absolute right-0 top-[calc(100%+10px)] w-52 rounded-2xl border border-white/10 bg-zinc-900/90 backdrop-blur-xl shadow-2xl shadow-black/60 p-2 z-50"
          >
            <div className="px-3 py-2 mb-1 border-b border-white/10">
              {user.Avatar && (
                <img
                  src={user.Avatar}
                  alt={user.Username ?? "Profile"}
                  referrerPolicy="no-referrer"
                  className="w-8 h-8 rounded-full object-cover mb-2"
                />
              )}
              <p className="text-white text-sm font-semibold truncate">
                {user.Username}
              </p>
            </div>

            <Logout />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
