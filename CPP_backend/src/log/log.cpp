#include "log.h"

Log* Log::getInstance() {
    static Log* instance = new Log();
    return instance;
}

void Log::Init(size_t level, const char* path, const char* suffix, size_t capacity) {
    level_ = level;
    path_ = path;
    suffix_ = suffix;
    is_open_ = true;
    is_async_ = capacity > 0;
    line_count_ = 0;

    if (is_async_) {
        q_ = std::make_unique<BlockQueue<std::string>>(capacity);
        write_thread_ = std::make_unique<std::thread>(Flush_all);
    }
    Init_file_();
}

void Log::Flush_all() {
    Log::getInstance()->Flush();
}

void Log::Write(size_t level, const char* format, ...) {
    Date date = tm_.get_cur_time();
    va_list va_list_;

    std::unique_lock<std::mutex> locker(mtx_);
    locker.unlock();

    char new_file[NAME_LEN_];
    char tail[36] = {0};
    snprintf(tail, 36, "%04d_%02d_%02d", int(date.year), int(date.month), int(date.day));

    if (to_day_ != date.day) {
        snprintf(new_file, NAME_LEN_ - 72, "%s/%s%s", path_, tail, suffix_);
        to_day_ = date.day;
        line_count_ = 0;
    } else {
        snprintf(new_file, NAME_LEN_ - 72, "%s/%s-%d%s", path_, tail, (line_count_ / MAX_LINE_NUMBER_), suffix_);
    }

    locker.lock();
    Flush();
    Close_file_();
    fp_ = fopen(new_file, "a");
    line_count_++;
    int n = snprintf(buffer_.Get_write_point(), 128, "%d-%02d-%02d %02d:%02d:%02d ",
                     int(date.year), int(date.month), int(date.day),
                     int(date.hour), int(date.minute), int(date.second));

    buffer_.Has_write(n);
    Append_log_level_(level);

    va_start(va_list_, format);
    int m = vsnprintf(buffer_.Get_write_point(), buffer_.Get_writable_size(), format, va_list_);
    va_end(va_list_);

    buffer_.Has_write(m);
    buffer_.Append("\n\0", 2);

    if (is_async_ && q_ && !q_->Full()) {
        q_->Push(buffer_.Get_read_all());
    } else {
        fputs(buffer_.Get_read_point_const(), fp_);
    }
    buffer_.Has_read_all();
}

void Log::Flush() {
    if (is_async_) {
        q_->Flush();
    }
    fflush(fp_);
}

size_t Log::Get_level() {
    return level_;
}

void Log::Set_level(size_t level) {
    level_ = level;
}

bool Log::Is_open() {
    return is_open_;
}

void Log::Append_log_level_(size_t level) {
    switch (level) {
        case 0: {
            buffer_.Append("type:[debug]: ", 14);
            break;
        }
        case 1: {
            buffer_.Append("type:[info]: ", 13);
            break;
        }
        case 2: {
            buffer_.Append("type:[warn]: ", 13);
            break;
        }
        case 3: {
            buffer_.Append("type:[error]: ", 14);
            break;
        }
        default: {
            buffer_.Append("type:[info] : ", 13);
            break;
        }
    }
}

void Log::Async_write_() {
    while (!q_->Empty()) {
        auto str = q_->Front();
        q_->Pop();
        std::lock_guard<std::mutex> lock(mtx_);
        fputs(str.c_str(), fp_);
    }
}

void Log::Init_file_() {
    char file_name[NAME_LEN_] = {0};
    Date date = tm_.get_cur_time();
    snprintf(file_name, NAME_LEN_ - 1, "%s/%04d_%02d_%02d%s",
             path_, int(date.year), int(date.month), int(date.day), suffix_);
    to_day_ = date.day;
    std::lock_guard<std::mutex> lock(mtx_);
    buffer_.Has_read_all();
    if (fp_) {
        Flush();
        fclose(fp_);
    }

    fp_ = fopen(file_name, "a");
    if (fp_ == nullptr) {
        mkdir(path_, 0777);
        fp_ = fopen(file_name, "a");
    }
}

void Log::Close_file_() {
    fclose(fp_);
}