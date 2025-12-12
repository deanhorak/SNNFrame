#include "snnfw/Main.h"

/**
 * @brief Entry point for the SNNFW application
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return Exit code (0 for success, non-zero for error)
 */
int main(int argc, char* argv[]) {
    // Create the main application instance
    snnfw::Main app(argc, argv);

    // Run the application and return its exit code
    return app.run();
}
