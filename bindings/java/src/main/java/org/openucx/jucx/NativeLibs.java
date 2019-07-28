/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.stream.Stream;

public class NativeLibs {
    private static final String UCM  = "libucm.so";
    private static final String UCS  = "libucs.so";
    private static final String UCT  = "libuct.so";
    private static final String UCP  = "libucp.so";
    private static final String JUCX = "libjucx.so";
    private static ClassLoader loader = NativeLibs.class.getClassLoader();
    private static String errorMessage = null;

    static {
        extractUCTLibs();   // UCT Transport
        loadLibrary(UCM);   // UCM library
        loadLibrary(UCS);   // UCS library
        loadLibrary(UCT);   // UCT library
        loadLibrary(UCP);   // UCP library
        loadLibrary(JUCX);  // JUCX native library
    }

    public static void load() {
        if (errorMessage != null) {
            throw new UnsatisfiedLinkError(errorMessage);
        }
    }

    /**
     * Tries to load the library, by extracting it from the current class jar.
     * If that fails, falls back on {@link System#loadLibrary(String)}.
     *
     * @param resourceName - library name to be extracted and loaded from the this current jar.
     */
    private static void loadLibrary(String resourceName) {
        // Search shared object on java classpath
        URL url = loader.getResource(resourceName);
        File file = null;
        try { // Extract shared object's content to a generated temp file
            file = extractResource(url);
        } catch (IOException ex) {
            errorMessage = "Native code library failed to extract URL: " + url;
            return;
        }

        if (file != null && file.exists()) {
            String filename = file.getAbsolutePath();
            try { // Load shared object to JVM
                System.load(filename);
            } catch (UnsatisfiedLinkError ex) {
                errorMessage = "Native code library failed to load: "
                    + resourceName;
            }

            file.deleteOnExit();
        }
    }

    /**
     * Extracts shared UCT transport.
     */
    private static void extractUCTLibs() {
        Path ucxTempFolder, ucxFolder;
        Stream<Path> uctLibs;
        final FileSystem fileSystem;
        try {
            createTempDir();
            ucxTempFolder = Files.createDirectory(Paths.get(tempDir.getPath(), "ucx"));
            ucxTempFolder.toFile().deleteOnExit();
            URI uri = NativeLibs.class.getClassLoader().getResource("ucx").toURI();
            if ("jar".equals(uri.getScheme())) {
                fileSystem = FileSystems.newFileSystem(uri, Collections.emptyMap(), null);
                ucxFolder = fileSystem.getPath("ucx");
            } else {
                ucxFolder = Paths.get(uri);
                fileSystem = null;
            }
            uctLibs = Files.walk(ucxFolder, 1);
        } catch (IOException ex) {
            errorMessage = "Failed to create temp directory";
            return;
        } catch (URISyntaxException e) {
            errorMessage = "Failed to find ucx resources";
            return;
        }

        uctLibs.forEach(filePath -> {
            if (!filePath.getFileName().toString().contains(".so")) {
                return;
            }
            FileOutputStream os = null;
            InputStream is = null;
            File out = new File(ucxTempFolder.toAbsolutePath().toString(),
                filePath.getFileName().toString());
            out.deleteOnExit();
            try {
                if (fileSystem != null) {
                    is = NativeLibs.class.getResourceAsStream(filePath.toString());
                } else {
                    is = new FileInputStream(filePath.toFile());
                }
                os = new FileOutputStream(out);
                copy(is, os);
            } catch (IOException ex) {
                errorMessage = "Failed to copy UCT lib: " + ex.getLocalizedMessage();
                return;
            } finally {
                closeQuietly(os);
                closeQuietly(is);
            }
        });

        if (fileSystem != null) {
            closeQuietly(fileSystem);
        }
    }

    /**
     * Extracts a resource into a temp directory.
     *
     * @param resourceURL - the URL of the resource to extract
     * @return the File object representing the extracted file
     * @throws IOException if fails to extract resource properly
     */
    private static File extractResource(URL resourceURL) throws IOException {
        InputStream is = resourceURL.openStream();
        if (is == null) {
            errorMessage = "Error extracting native library content";
            return null;
        }

        try {
            createTempDir();
        } catch (IOException ex) {
            errorMessage = "Failed to create temp directory";
            return null;
        }

        File file = new File(tempDir,
            new File(resourceURL.getPath()).getName());
        file.deleteOnExit();
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

    /**
     * Temporary directory set and returned by {@link #createTempDir()}.
     */
    static File tempDir = null;

    /**
     * Creates a new temp directory in default temp files directory.
     * Directory will be represented by {@link #tempDir}.
     */
    private static void createTempDir() throws IOException {
        if (tempDir == null) {
            Path tmp = Files.createTempDirectory("jucx");
            tempDir = tmp.toFile();
            tempDir.deleteOnExit();
        }
    }

    /**
     * Helper function to copy an InputStream into an OutputStream.
     */
    private static void copy(InputStream is, OutputStream os)
        throws IOException {
        if (is == null || os == null) {
            return;
        }
        byte[] buffer = new byte[1024];
        int length = 0;
        while ((length = is.read(buffer)) != -1) {
            os.write(buffer, 0, length);
        }
    }

    /**
     * Helper function to close InputStream or OutputStream in a quiet way
     * which hides the exceptions.
     */
    private static void closeQuietly(Closeable closable) {
        if (closable == null) {
            return;
        }
        try {
            closable.close();
        } catch (IOException ex) {
            // No logging in this 'Quiet Close' method
        }
    }
}
