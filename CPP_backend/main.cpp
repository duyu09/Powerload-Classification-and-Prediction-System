#include "src/server/webserver.h"

int main() {
    WebServer server(5000, 3, 60000, false, "./powerload.db", 12, 6, true, 0, 0);
    server.Start();
} 