cmake_minimum_required(VERSION 3.5.0)
project(webserver VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(BUFFER src/buffer/buffer.cpp)
set(HTTP src/http/httpconnect.cpp src/http/httprequest.cpp src/http/httpresponse.cpp)
set(LOG src/log/log.cpp)
set(POOL src/pool/threadpool.cpp)
set(SERVER src/server/epoller.cpp src/server/webserver.cpp)
set(TIMER src/timer/timer.cpp)
set(SQLITE src/sqlite/sqlite3.c)
set(DATABASE src/database/database.cpp)
set(SRC ${BUFFER} ${HTTP} ${LOG} ${POOL} ${SERVER} ${TIMER} ${SQLITE} ${DATABASE})

find_package(Threads REQUIRED)
add_executable(webserver main.cpp ${SRC})
target_link_libraries(webserver Threads::Threads)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
