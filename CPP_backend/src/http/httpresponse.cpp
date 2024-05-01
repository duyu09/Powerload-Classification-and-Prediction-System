#include "httpresponse.h"

HttpResponse::HttpResponse() {
    code_ = -1;
    isKeepAlive_ = false;
};

void HttpResponse::Init(size_t code, bool isKeepAlive) {
    code_ = code;
    isKeepAlive_ = isKeepAlive;
}

void HttpResponse::MakeResponse(Buffer& buff, HttpRequest request) {
    if(WEB_SITE.find(request.path()) == WEB_SITE.end()) {
        code_ = 404;
        std::string body = "";
        std::string status;
        body += "<html><title>Error</title>";
        body += "<body bgcolor=\"ffffff\">";
        if(CODE_STATUS.count(code_) == 1) {
            status = CODE_STATUS.find(code_)->second;
        } else {
            status = "Bad Request";
        }
        body += std::to_string(code_) + " : " + status  + "\n";
        body += "<p>";
        body += + "You access an error website!";
        body += "</p>";

        buff.Append("Content-length: " + std::to_string(body.size()) + "\r\n\r\n");
        buff.Append(body);
    } else {
        std::string status;
        if(CODE_STATUS.count(code_) == 1) {
            status = CODE_STATUS.find(code_)->second;
        }
        else {
            code_ = 400;
            status = CODE_STATUS.find(400)->second;
            std::string body = "";
            std::string status;
            body += "<html><title>Error</title>";
            body += "<body bgcolor=\"ffffff\">";
            body += std::to_string(code_) + " : " + status  + "\n";
            body += "<p>";
            body += + "You access an error website!";
            body += "</p>";
            return ;
        }

        buff.Append("HTTP/1.1 " + std::to_string(code_) + " " + status + "\r\n");

        buff.Append("Connection: ");
        if(isKeepAlive_) {
            buff.Append("keep-alive\r\n");
            buff.Append("keep-alive: max=6, timeout=120\r\n");
        } else{
            buff.Append("close\r\n");
        }
        buff.Append("Content-type: application/json\r\n");
        buff.Append("\r\n");

        auto it = WEB_SITE.find(request.path());
        if(it->second == "LOGIN") {
            login_(buff, request);
        } else if(it->second == "REGISTER") {
            register_(buff, request);
        } else if(it->second == "GET") {
            getdata_(buff, request);
        }
    }
}

void HttpResponse::login_(Buffer &buff, HttpRequest request) {
    DataBase* db = DataBase::GetInstance();
    std::string ps = std::move(db->query_user_password(request.parameter["phone_number"]));
    if(ps == request.parameter["password"]) {
        std::string token = GUIDGenerator::GenerateFakeGUID();
        std::ostringstream osr;
        osr << "{\r\n\"code\" : \"0\",\r\n \"token\" : ";
        osr << "\"" << token << "\"" << "}";
        osr << "\r\n";
        buff.Append(osr.str());
        match_[request.parameter["phone_number"]] = token;
    } else {
        std::ostringstream osr;
        osr << "{\r\n\"code\" : \"1\",\r\n \"token\" : ";
        osr << "\"" << "\"" << "}";
        osr << "\r\n";
        buff.Append(osr.str());
    }
}

void HttpResponse::register_(Buffer &buff, HttpRequest request) {
    DataBase* db = DataBase::GetInstance();
    std::string ps = std::move(db->query_user_password(request.parameter["phone_number"]));
    if(ps == request.parameter["password"]) {
        std::ostringstream osr;
        osr << "{\r\n\"code\" : 1,\r\n \"description\" : ";
        osr << "\"电话号码已注册\"\r\n}";
        buff.Append(osr.str());
    } else if(db->insert_user(request.parameter["password"], request.parameter["phone_number"], 
            request.parameter["room"].c_str())) {
        std::ostringstream osr;
        osr << "{\r\n\"code\" : 0,\r\n \"description\" : ";
        osr << "\"注册成功\"\r\n}";
        buff.Append(osr.str());
    } else {
        std::ostringstream osr;
        osr << "{\r\n\"code\" : 1,\r\n \"description\" : ";
        osr << "\"其他错误\"\r\n}";
        buff.Append(osr.str());
    }
}

void HttpResponse::getdata_(Buffer &buff, HttpRequest request) {
    DataBase* ds = DataBase::GetInstance();
    if(request.parameter.find("token") == request.parameter.end() || request.parameter["token"] != match_[request.parameter["phone_number"]]) {
        std::ostringstream osr;
        osr << "{\r\n\"code\" : 1,\r\n \"description\" : ";
        osr << "\"token不存在或token不匹配\"\r\n}";
        buff.Append(osr.str());
    } else {
        auto id = ds->query_user_id(request.parameter["phone_number"]);
        if(id == -1) {
            std::ostringstream osr;
            osr << "{\r\n\"code\" : 1,\r\n \"description\" : ";
            osr << "\"用户未找到\"\r\n}";
            buff.Append(osr.str());
        } else {
            auto real_time = std::move(ds->query_real(id));
            auto prediction = std::move(ds->query_prediction(id));
            auto label = std::move(ds->query_legal(id));
            std::ostringstream osr;
            osr << "{\r\n\"code\" : 0,\r\n \"description\" : ";
            osr << "\"读取成功\"\r\n,";
            osr << "\"real_time\" : [";
            while(!real_time.empty()) {
                auto u = real_time.front(); real_time.pop_front();
                osr << std::to_string(u);
                if(!real_time.empty()) osr << ",";
            }
            osr << "],\r\n";
            osr << "\"prediction\" : [";
            while(!prediction.empty()) {
                auto u = prediction.front(); prediction.pop_front();
                osr << std::to_string(u);
                if(!prediction.empty()) osr << ",";
            }
            osr << "],\r\n";
            osr << "\"label\" : \"";
            osr << label;
            osr << "\"}";
            osr << "\r\n";
            buff.Append(osr.str()); 
        }
    }
}