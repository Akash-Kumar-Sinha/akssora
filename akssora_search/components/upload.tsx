"use client";
import api from "@/lib/api";
import { BACKEND_URL } from "@/lib/constant";
import React, { useState } from "react";
import { FileUpload } from "@/components/ui/file-upload";
import { SlideButton } from "./ui/slide-button";

type UploadType = "image" | "video";
const imageExtensions = new Set([".jpg", ".jpeg", ".png"]);
const videoExtensions = new Set([".mp4"]);

type FileType = {
  file: File | null;
  fileType: UploadType | null;
  maxImageSizeMB: number;
  maxVideoSizeMB: number;
};

const Upload = () => {
  const [value, setValue] = useState<File[]>([]);
  const [file, setFile] = useState<FileType>({
    file: null,
    fileType: null,
    maxImageSizeMB: 5,
    maxVideoSizeMB: 100,
  });

  const handleSetValue = (files: File[]) => {
    setValue(files);

    const selectedFile = files[0];
    if (!selectedFile) return;

    const fileName = selectedFile.name.toLowerCase();
    const extension = fileName.slice(fileName.lastIndexOf("."));

    let fileType: UploadType | null = null;
    if (imageExtensions.has(extension)) {
      fileType = "image";
    } else if (videoExtensions.has(extension)) {
      fileType = "video";
    }

    if (!fileType) {
      alert("Only jpg, jpeg, png, and mp4 files are allowed.");
      return;
    }

    setFile((prev) => ({ ...prev, file: selectedFile, fileType }));
  };

  const handleUpload = async () => {
    if (!file.file) return;

    const formData = new FormData();
    formData.append("file", file.file);

    const res = await api.post(`${BACKEND_URL}/app/upload`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
      withCredentials: true,
    });
    console.log(res.data);
  };

  return (
    <div className="flex justify-center w-full">
      <div className="flex justify-between w-3xl gap-2">
        <FileUpload value={value} setValue={handleSetValue} maxFiles={1} />
        <SlideButton onClick={handleUpload}>Upload</SlideButton>
      </div>
    </div>
  );
};

export default Upload;
