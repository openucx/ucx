/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import java.io.Closeable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;

public class LoadLibrary {
    static String errorMessage = null;

    /**
     * Tries to load the library, by extracting it from the current class jar.
     * If that fails, falls back on {@link System#loadLibrary(String)}.
     *
     * @param String
     *            to the library name to be extracted and loaded from the this
     *            current jar
     */
    public static void loadLibrary(String resourceName) {
        ClassLoader loader = LoadLibrary.class.getClassLoader();

        // Search shared object on java classpath
        URL url = loader.getResource(resourceName);
        File file = null;
        try { // Extract shared object's content to a generated temp file
            file = extractResource(url);
        } catch (IOException e) {
            errorMessage = "Native code library failed to extract URL: " + url;
            return;
        }

        if (file != null && file.exists()) {
            String filename = file.getAbsolutePath();
            try { // Load shared object to JVM
                System.load(filename);
            } catch (UnsatisfiedLinkError e) {
                errorMessage = "Native code library failed to load: "
                        + resourceName;
            }

            file.deleteOnExit();
        }
    }

    /**
     * Extracts a resource into a temp directory.
     *
     * @param resourceURL
     *            the URL of the resource to extract
     * @return the File object representing the extracted file
     * @throws IOException
     *             if fails to extract resource properly
     */
    public static File extractResource(URL resourceURL) throws IOException {
        InputStream is = resourceURL.openStream();
        if (is == null) {
            errorMessage = "Error extracting native library content";
            return null;
        }

        try {
            createTempDir();
        } catch (IOException e) {
            errorMessage = "Failed to create temp directory";
            return null;
        }

        File file = new File(tempDir,
                new File(resourceURL.getPath()).getName());
        FileOutputStream os = null;
        try {
            os = new FileOutputStream(file);
            copy(is, os);
        } finally {
            closeQuietly(os);
            closeQuietly(is);
        }
        return file;
    }

    /** Temporary directory set and returned by {@link #createTempDir()}. */
    static File tempDir = null;

    /**
     * Creates a new temp directory in default temp files directory.
     * Directory will be represented by {@link #tempDir}.
     */
    public static void createTempDir() throws IOException {
        if (tempDir == null) {
            Path p = Files.createTempDirectory("jucx");
            tempDir = p.toFile();
            tempDir.deleteOnExit();
        }
    }

    /**
     * Helper function to copy an InputStream into an OutputStream
     */
    public static void copy(InputStream is, OutputStream os)
            throws IOException {
        if (is == null || os == null)
            return;
        byte[] buffer = new byte[1024];
        int length = 0;
        while ((length = is.read(buffer)) != -1) {
            os.write(buffer, 0, length);
        }
    }

    /**
     * Helper function to close InputStream or OutputStream in a quiet way
     * which hides the exceptions
     */
    public static void closeQuietly(Closeable c) {
        if (c == null)
            return;
        try {
            c.close();
        } catch (IOException e) {
            // No logging in this 'Quiet Close' method
        }
    }
}
