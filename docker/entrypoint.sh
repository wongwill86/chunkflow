#!/bin/bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or fallback to 9001
# inspired by https://denibertovic.com/posts/handling-permissions-with-docker-volumes/

USER_ID=${LOCAL_USER_ID:-9001}
CHUNKFLOW_USER=${CHUNKFLOW_USER}
CHUNKFLOW_HOME=${CHUNKFLOW_HOME}

echo "Running with UID : $USER_ID"
groupadd -r ${CHUNKFLOW_USER}
useradd -r -d ${CHUNKFLOW_HOME} -g ${CHUNKFLOW_USER} -s /bin/bash ${CHUNKFLOW_USER} -u $USER_ID
chown -R ${CHUNKFLOW_USER}: ${CHUNKFLOW_HOME}

exec /usr/local/bin/gosu ${CHUNKFLOW_USER} "$@"
