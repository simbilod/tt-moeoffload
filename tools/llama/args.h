// The producer tools' shared helpers: their own options taken out of argv before llama.cpp's parser sees the rest,
// and a JSON string field of one corpus line (the corpus is json.dumps output, one object per line).
#pragma once

#include <cstring>
#include <string>

static std::string take_arg(int & argc, char ** argv, const char * name, const char * def = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            std::string v = argv[i + 1];
            for (int j = i; j + 2 < argc; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            return v;
        }
    }
    return def;
}

static std::string json_str(const std::string & line, const char * key) {
    const std::string k = std::string("\"") + key + "\": \"";
    size_t p = line.find(k);
    if (p == std::string::npos) {
        return "";
    }
    std::string out;
    for (p += k.size(); p < line.size() && line[p] != '"'; ++p) {
        if (line[p] != '\\') {
            out += line[p];
            continue;
        }
        const char c = line[++p];
        switch (c) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'u': {
                const unsigned cp = std::stoul(line.substr(p + 1, 4), nullptr, 16);
                p += 4;
                unsigned code = cp;
                if (cp >= 0xD800 && cp <= 0xDBFF && line.compare(p + 1, 2, "\\u") == 0) {
                    const unsigned lo = std::stoul(line.substr(p + 3, 4), nullptr, 16);
                    code = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    p += 6;
                }
                if (code < 0x80) { out += (char) code; }
                else if (code < 0x800) { out += (char) (0xC0 | (code >> 6)); out += (char) (0x80 | (code & 0x3F)); }
                else if (code < 0x10000) { out += (char) (0xE0 | (code >> 12)); out += (char) (0x80 | ((code >> 6) & 0x3F)); out += (char) (0x80 | (code & 0x3F)); }
                else { out += (char) (0xF0 | (code >> 18)); out += (char) (0x80 | ((code >> 12) & 0x3F)); out += (char) (0x80 | ((code >> 6) & 0x3F)); out += (char) (0x80 | (code & 0x3F)); }
                break;
            }
            default: out += c;
        }
    }
    return out;
}
