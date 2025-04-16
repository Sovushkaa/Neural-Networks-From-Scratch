#include "Except.h"
#include <exception>
#include <stdexcept>
#include <iostream>

namespace except {
    void react() {
        try {
            throw;
        } catch (const std::runtime_error &e) {
            std::cerr << "Error: Runtime error: " << e.what() << std::endl;
        } catch (const std::invalid_argument &e) {
            std::cerr << "Error: Invalid argument: " << e.what() << std::endl;
        } catch (const std::out_of_range &e) {
            std::cerr << "Error: Out of range: " << e.what() << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Error: Unknown error " << std::endl;
        }
    }
}
