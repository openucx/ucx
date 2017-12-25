<h1>JUCX</h1>

JUCX is a Java API over UCP (UCX protocol).</br>
See more about UCX at: https://github.com/openucx/ucx

# Building JUCX
Building the source requires [Apache Maven](http://maven.apache.org/) and [GNU/autotools](http://www.gnu.org/software/autoconf/autoconf.html) and Java version 8 or higher.</br>
Java binding will be built by default, but it is recommended to execute the following steps:
1. export JAVA_HOME=\<path-to-java\>.
2. When running UCX's "configure" add "--with-java" flag, i.e. "shell$ ./configure --with-java".
