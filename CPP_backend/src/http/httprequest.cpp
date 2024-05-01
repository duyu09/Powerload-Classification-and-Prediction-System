#include "httprequest.h"


void HttpRequest::Init() {
    method_ = path_ = version_ = body_ = "";
    state_ = REQUEST_LINE;
    header_.clear();
}

bool HttpRequest::IsKeepAlive() const {
    if(header_.count("Connection") == 1) {
        return header_.find("Connection")->second == "keep-alive" && version_ == "1.1";
    }
    return false;
}

bool HttpRequest::parse(Buffer& buff) {
    const char C[] = "\r\n";
    if(buff.Get_readable_size() <= 0) {
        return false;
    }
    while(buff.Get_readable_size() && state_ != FINISH) {
        const char* lineEnd = std::search(buff.Get_read_point_const(), buff.Get_write_point_const(), C, C + 2);
        std::string line(buff.Get_read_point_const(), lineEnd);
        switch(state_) {
            case REQUEST_LINE:
                if(!ParseRequestLine_(line)) {
                    return false;
                }
                break;
            case HEADERS:
                ParseHeader_(line);
                if(buff.Get_readable_size() <= 2) {
                    state_ = FINISH;
                }
                break;
            case BODY:
                body_ += line;
                break;
            default:
                break;
        }
        if(lineEnd == buff.Get_write_point_const()) { break; }
        buff.Has_read_until(lineEnd + 2);
    }
    ParseBody_();
    return true;
}

bool HttpRequest::ParseRequestLine_(const std::string& line) {
    std::regex patten("^([^ ]*) ([^ ]*) HTTP/([^ ]*)$");
    std::smatch subMatch;
    if(regex_match(line, subMatch, patten)) {
        method_ = subMatch[1];
        path_ = subMatch[2];
        version_ = subMatch[3];
        state_ = HEADERS;
        return true;
    }
    LOG_ERROR("RequestLine Error");
    return false;
}

void HttpRequest::ParseHeader_(const std::string& line) {
    std::regex patten("^([^:]*): ?(.*)$");
    std::smatch subMatch;
    if(regex_match(line, subMatch, patten)) {
        header_[subMatch[1]] = subMatch[2];
    } else {
        state_ = BODY;
    }
}

void HttpRequest::ParseBody_() {
    std::string::size_type pos = 0;
    while ((pos = body_.find("\"", pos)) != std::string::npos) {
        std::string::size_type end = body_.find("\"", pos + 1);
        if (end == std::string::npos) {
            break;
        }
        std::string k = body_.substr(pos + 1, end - pos - 1);
        pos = body_.find("\"", end + 1);
        if (pos == std::string::npos) {
            break;
        }
        end = body_.find("\"", pos + 1);
        if (end == std::string::npos) {
            break;
        }
        std::string v = body_.substr(pos + 1, end - pos - 1);
        parameter[k] = v;
        pos = end + 1;
    }
    state_ = FINISH;
}

std::string HttpRequest::path() const{
    return path_;
}

std::string& HttpRequest::path(){
    return path_;
}