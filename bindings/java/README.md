<h1>JUCX</h1>

JUCX is a Java API over UCP (UCX protocol).</br>
See more about UCX at: https://github.com/openucx/ucx

# Building JUCX
Building the source requires [Apache Maven](http://maven.apache.org/) and [GNU/autotools](http://www.gnu.org/software/autoconf/autoconf.html) and Java version 8 or higher.</br>
Java binding will be built by default, but it is recommended to execute the following steps:
1. export JAVA_HOME=\<path-to-java\>.
2. When running UCX's "configure" add "--with-java" flag, i.e. "shell$ ./configure --with-java".

# Running against an uninstalled UCX build

Maven tests use optional UCX modules from the configured build tree through
`UCX_MODULE_DIR`. Run `mvn test` from that build's `bindings/java` directory
after building the native libraries.

For standalone Java launches using the build's native resources, set the same
module directory explicitly:

```sh
UCX_MODULE_DIR=/absolute/path/to/ucx-build/modules java <application arguments>
```

Keep the modules in the build tree so their SONAME symlinks remain intact.
When reusing an older build tree that copied modules into `resources/ucx`,
run `mvn clean test` once from its `bindings/java` directory before launching
Java again. This removes stale module copies from both `resources` and the
Maven test output, which otherwise take precedence over `UCX_MODULE_DIR`,
then recreates the native resources needed for the launch.
