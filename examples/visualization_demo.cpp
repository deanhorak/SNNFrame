#include "snnfw/VisualizationManager.h"
#include "snnfw/Camera.h"
#include "snnfw/ShaderManager.h"
#include "snnfw/GeometryRenderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <iostream>

using namespace snnfw;

int main() {
    try {
        // Create visualization manager
        VisualizationManager vizManager(1280, 720, "SNNFW Visualization Demo - Phase 1");
        vizManager.setTargetFPS(60);
        vizManager.enableVSync(true);
        
        // Create camera
        Camera camera(glm::vec3(0, 2, 5), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        
        // Create shader manager
        ShaderManager shaderManager;
        GLuint shader = shaderManager.loadShader("basic", "shaders/basic.vert", "shaders/basic.frag");
        if (shader == 0) {
            std::cerr << "Failed to load shaders!" << std::endl;
            return 1;
        }
        
        // Create geometry renderer
        GeometryRenderer geometryRenderer;
        
        // Demo state
        float rotation = 0.0f;
        float rotationSpeed = 1.0f;
        glm::vec4 cubeColor(0.2f, 0.5f, 1.0f, 1.0f);
        glm::vec4 sphereColor(1.0f, 0.3f, 0.2f, 1.0f);
        bool showCube = true;
        bool showSphere = true;
        bool autoRotate = true;
        
        // Camera control state
        bool mousePressed = false;
        double lastMouseX = 0.0;
        double lastMouseY = 0.0;
        
        // Set update callback
        vizManager.setUpdateCallback([&](double deltaTime) {
            // Auto-rotate
            if (autoRotate) {
                rotation += rotationSpeed * deltaTime;
            }
            
            // Camera controls
            GLFWwindow* window = vizManager.getWindow();
            
            // Mouse orbit
            double mouseX, mouseY;
            glfwGetCursorPos(window, &mouseX, &mouseY);
            
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                if (mousePressed) {
                    float deltaX = (mouseX - lastMouseX) * 0.01f;
                    float deltaY = (mouseY - lastMouseY) * 0.01f;
                    camera.orbit(deltaX, -deltaY);
                }
                mousePressed = true;
            } else {
                mousePressed = false;
            }
            
            lastMouseX = mouseX;
            lastMouseY = mouseY;
            
            // Keyboard pan
            float panSpeed = 2.0f * deltaTime;
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                camera.pan(0, panSpeed);
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                camera.pan(0, -panSpeed);
            }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                camera.pan(-panSpeed, 0);
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                camera.pan(panSpeed, 0);
            }
            
            // Mouse scroll zoom
            // (This would need a scroll callback, simplified for now)
        });
        
        // Set render callback
        vizManager.setRenderCallback([&](double deltaTime) {
            // Use shader
            shaderManager.useShader("basic");
            
            // Set view and projection matrices
            glm::mat4 view = camera.getViewMatrix();
            glm::mat4 projection = camera.getProjectionMatrix(vizManager.getAspectRatio());
            
            shaderManager.setUniform("view", view);
            shaderManager.setUniform("projection", projection);
            shaderManager.setUniform("viewPos", camera.getPosition());
            shaderManager.setUniform("lightPos", glm::vec3(5.0f, 5.0f, 5.0f));
            
            // Render cube
            if (showCube) {
                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, glm::vec3(-1.5f, 0.0f, 0.0f));
                model = glm::rotate(model, rotation, glm::vec3(0.0f, 1.0f, 0.0f));
                model = glm::rotate(model, rotation * 0.5f, glm::vec3(1.0f, 0.0f, 0.0f));
                
                shaderManager.setUniform("model", model);
                shaderManager.setUniform("color", cubeColor);
                
                geometryRenderer.renderCube();
            }
            
            // Render sphere
            if (showSphere) {
                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, glm::vec3(1.5f, 0.0f, 0.0f));
                model = glm::rotate(model, rotation * 0.7f, glm::vec3(0.0f, 1.0f, 0.0f));
                model = glm::scale(model, glm::vec3(1.2f));
                
                shaderManager.setUniform("model", model);
                shaderManager.setUniform("color", sphereColor);
                
                geometryRenderer.renderSphere();
            }
            
            // ImGui controls
            ImGui::Begin("Demo Controls");
            
            ImGui::Text("Phase 1: Foundation");
            ImGui::Separator();
            
            ImGui::Checkbox("Auto Rotate", &autoRotate);
            ImGui::SliderFloat("Rotation Speed", &rotationSpeed, 0.0f, 5.0f);
            
            ImGui::Separator();
            ImGui::Checkbox("Show Cube", &showCube);
            if (showCube) {
                ImGui::ColorEdit4("Cube Color", &cubeColor.r);
            }
            
            ImGui::Checkbox("Show Sphere", &showSphere);
            if (showSphere) {
                ImGui::ColorEdit4("Sphere Color", &sphereColor.r);
            }
            
            ImGui::Separator();
            ImGui::Text("Camera Controls:");
            ImGui::BulletText("Left Mouse: Orbit");
            ImGui::BulletText("WASD: Pan");
            ImGui::BulletText("ESC: Exit");
            
            ImGui::Separator();
            ImGui::Text("Camera Position: (%.1f, %.1f, %.1f)", 
                        camera.getPosition().x, 
                        camera.getPosition().y, 
                        camera.getPosition().z);
            ImGui::Text("Camera Distance: %.1f", camera.getDistance());
            
            ImGui::End();
        });
        
        // Run main loop
        std::cout << "Starting visualization demo..." << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  - Left Mouse: Orbit camera" << std::endl;
        std::cout << "  - WASD: Pan camera" << std::endl;
        std::cout << "  - ESC: Exit" << std::endl;
        
        vizManager.run();
        
        std::cout << "Demo finished successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

