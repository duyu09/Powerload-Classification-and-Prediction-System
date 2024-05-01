#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sqlite3.h>

class HttpServer {
public:
    HttpServer() : conn(nullptr) {
        int rc = sqlite3_open("powerload.db", &conn);
        if (rc) {
            std::cerr << "Can't open database: " << sqlite3_errmsg(conn) << std::endl;
            sqlite3_close(conn);
            exit(EXIT_FAILURE);
        }
    }

    ~HttpServer() {
        sqlite3_close(conn);
    }

    void start() {
        // Register routes
        registerRoute("/api/test", HttpMethod::GET, &HttpServer::appTest);
        registerRoute("/api/login", HttpMethod::POST, &HttpServer::login);
        registerRoute("/api/register", HttpMethod::POST, &HttpServer::registerUser);
        registerRoute("/api/getdata", HttpMethod::POST, &HttpServer::getData);

        // Start server
        // Your server logic here
    }

private:
    sqlite3* conn;

    enum class HttpMethod { GET, POST };

    typedef void (HttpServer::*RouteHandler)(const std::unordered_map<std::string, std::string>&);

    struct Route {
        std::string path;
        HttpMethod method;
        RouteHandler handler;
    };

    std::vector<Route> routes;

    void registerRoute(const std::string& path, HttpMethod method, RouteHandler handler) {
        routes.push_back({path, method, handler});
    }

    void appTest(const std::unordered_map<std::string, std::string>& params) {
        // Handle test route
        std::cout << "Hello, World!" << std::endl;
    }

    void login(const std::unordered_map<std::string, std::string>& params) {
        std::string phoneNumber = params.at("phone_number");
        std::string passwordMd5 = params.at("password");

        // Execute SQL query to authenticate user
        // Handle login logic here
    }

    void registerUser(const std::unordered_map<std::string, std::string>& params) {
        std::string phoneNumber = params.at("phone_number");
        std::string passwordMd5 = params.at("password");
        std::string room = params.at("room");

        // Execute SQL query to register user
        // Handle user registration logic here
    }

    void getData(const std::unordered_map<std::string, std::string>& params) {
        std::string phoneNumber = params.at("phone_number");
        std::string token = params.at("token");

        // Execute SQL query to fetch data
        // Handle data retrieval logic here
    }
};

int main() {
    HttpServer server;
    server.start();
    return 0;
}
