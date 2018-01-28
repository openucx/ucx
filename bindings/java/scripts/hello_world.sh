#!/bin/bash

TOP_DIR="$(cd "$(dirname "$0")"/../"" && pwd)"

# Check arguments
if [ -z $1 ]; then
    echo -e "$0 ERROR: Missig first parameter. Should be 'server' or 'client'."
    exit 1
fi

classpath=.:../jucx-0.0.1-SNAPSHOT.jar

if [[ "server" == $1 ]]; then
	cd $TOP_DIR/target/test-classes/ && java -cp $classpath org.ucx.jucx.examples.helloworld.HelloServer "${@:2}"
elif [[ "client" == $1 ]]; then
	cd $TOP_DIR/target/test-classes/ && java -cp $classpath org.ucx.jucx.examples.helloworld.HelloClient "${@:2}"
else
	echo -e "$0 ERROR: First parameter should be 'server' or 'client'."
	exit 1
fi

