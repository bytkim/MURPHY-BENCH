// Offline llama.cpp vocabulary/tokenizer control. No weights or inference loaded.
#include "llama.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], params);
    if (!model) return 3;
    const auto * vocab = llama_model_get_vocab(model);
    nlohmann::json result, inputs;
    std::ifstream file(argv[2]);
    file >> inputs;
    result["pieces"] = nlohmann::json::array();
    for (int id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
        std::vector<char> piece(256);
        int n = llama_token_to_piece(vocab, id, piece.data(), piece.size(), 0, false);
        if (n < 0) {
            piece.resize(-n);
            n = llama_token_to_piece(vocab, id, piece.data(), piece.size(), 0, false);
        }
        if (n < 0) throw std::runtime_error("piece size mismatch");
        std::string hex;
        for (int j = 0; j < n; ++j) {
            unsigned char byte = piece[j];
            hex += "0123456789abcdef"[byte >> 4];
            hex += "0123456789abcdef"[byte & 15];
        }
        result["pieces"].push_back(hex);
    }
    result["token_ids"] = nlohmann::json::array();
    result["nfc_token_ids"] = nlohmann::json::array();
    for (const auto & row : inputs.at("records")) {
        for (const auto & field : {"text", "nfc_text"}) {
            std::string text = row.value(field, row.at("text").get<std::string>());
            int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, false);
            std::vector<llama_token> ids(n < 0 ? -n : n);
            n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), false, false);
            if (n < 0) throw std::runtime_error("token size mismatch");
            ids.resize(n);
            result[std::string(field) == "text" ? "token_ids" : "nfc_token_ids"].push_back(ids);
        }
    }
    llama_model_free(model);
    std::cout << result.dump() << '\n';
}
