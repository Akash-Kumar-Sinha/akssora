"use client";

import React from "react";
import useUser from "@/lib/hook/useUser";
import Upload from "@/components/upload";

const Layout = ({ children }: { children: React.ReactNode }) => {
  const { user, loading } = useUser();
  if (loading) return <div>Loading...</div>;

  if (!user) {
    return <div>Redirecting...</div>;
  }

  return (
    <div className="relative w-full min-h-screen flex justify-center">
      {children}
      <div className="absolute bottom-0 w-full">
        <Upload />
      </div>
    </div>
  );
};

export default Layout;
