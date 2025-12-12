#include "snnfw/VisualizationManager.h"
#include "snnfw/Camera.h"
#include "snnfw/ShaderManager.h"
#include "snnfw/NetworkDataAdapter.h"
#include "snnfw/LayoutEngine.h"
#include "snnfw/NetworkGraphRenderer.h"
#include "snnfw/Datastore.h"
#include "snnfw/NetworkInspector.h"
#include "snnfw/ActivityMonitor.h"
#include "snnfw/NetworkBuilder.h"
#include "snnfw/NeuralObjectFactory.h"
#include <imgui.h>
#include <iostream>
#include <memory>

using namespace snnfw;

int main() {
    std::cout << "SNNFW Network Visualization Demo" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Create datastore
    Datastore datastore("./network_viz_db", 100000);

    // Create factory
    NeuralObjectFactory factory;

    // Create network inspector and activity monitor
    NetworkInspector inspector;
    ActivityMonitor activityMonitor(datastore);

    // Build a simple test network - just create a single cluster with neurons
    std::cout << "Building test network..." << std::endl;

    // Create a simple cluster with neurons
    auto cluster = factory.createCluster();

    // Create 20 neurons
    std::vector<uint64_t> neuronIds;
    for (int i = 0; i < 20; ++i) {
        auto neuron = factory.createNeuron(50.0, 0.95, 20);

        // Create axon and dendrite for the neuron
        auto axon = factory.createAxon(neuron->getId());
        auto dendrite = factory.createDendrite(neuron->getId());

        neuron->setAxonId(axon->getId());
        neuron->addDendrite(dendrite->getId());

        // Store in datastore
        datastore.put(neuron);
        datastore.put(axon);
        datastore.put(dendrite);

        neuronIds.push_back(neuron->getId());
    }

    // Add neurons to cluster
    for (uint64_t neuronId : neuronIds) {
        cluster->addNeuron(neuronId);
    }
    datastore.put(cluster);

    std::cout << "Created cluster with " << neuronIds.size() << " neurons" << std::endl;
    std::cout << "Cluster ID: " << cluster->getId() << std::endl;
    
    // Create visualization manager
    VisualizationManager vizManager(1280, 720, "SNNFW Network Visualization");
    vizManager.setDatastore(&datastore);
    vizManager.setNetworkInspector(&inspector);
    vizManager.setActivityMonitor(&activityMonitor);
    
    // Create camera
    Camera camera(glm::vec3(0, 20, 50), glm::vec3(0, 0, 0));
    
    // Create shader manager
    ShaderManager shaderManager;
    
    // Create network data adapter
    NetworkDataAdapter adapter(datastore, inspector, &activityMonitor);
    
    // Extract network from the cluster
    std::cout << "Extracting network structure..." << std::endl;

    if (!adapter.extractHierarchy(cluster->getId(), "Cluster")) {
        std::cerr << "Failed to extract network!" << std::endl;
        return 1;
    }
    
    std::cout << "Extracted " << adapter.getNeuronCount() << " neurons" << std::endl;
    std::cout << "Extracted " << adapter.getSynapseCount() << " synapses" << std::endl;
    
    // Create layout engine and compute layout
    std::cout << "Computing layout..." << std::endl;
    LayoutEngine layoutEngine;
    LayoutConfig layoutConfig;
    layoutConfig.algorithm = LayoutAlgorithm::HIERARCHICAL_TREE;
    layoutConfig.neuronSpacing = 2.0f;
    layoutConfig.clusterSpacing = 10.0f;
    layoutConfig.layerSpacing = 20.0f;
    layoutConfig.columnSpacing = 30.0f;
    layoutConfig.centerLayout = true;
    
    layoutEngine.setProgressCallback([](float progress) {
        std::cout << "Layout progress: " << (progress * 100) << "%\r" << std::flush;
    });
    
    if (!layoutEngine.computeLayout(adapter, layoutConfig)) {
        std::cerr << "Failed to compute layout!" << std::endl;
        return 1;
    }
    
    std::cout << std::endl << "Layout computed successfully!" << std::endl;
    
    // Update synapse positions
    adapter.updateSynapsePositions();
    
    // Create network graph renderer
    NetworkGraphRenderer networkRenderer(shaderManager);
    if (!networkRenderer.initialize()) {
        std::cerr << "Failed to initialize network renderer!" << std::endl;
        return 1;
    }
    
    std::cout << "Network renderer initialized!" << std::endl;
    
    // Rendering configuration
    RenderConfig renderConfig;
    renderConfig.mode = RenderMode::NEURONS_AND_SYNAPSES;
    renderConfig.renderNeurons = true;
    renderConfig.renderSynapses = true;
    renderConfig.neuronUseInstancing = true;
    renderConfig.neuronBaseRadius = 0.5f;
    renderConfig.synapseBaseThickness = 0.1f;
    renderConfig.enableLighting = true;
    renderConfig.frustumCulling = false;  // Disable for small network
    renderConfig.distanceCulling = false;
    
    // Camera control state
    bool leftMouseDown = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    
    // Set update callback
    vizManager.setUpdateCallback([&](double deltaTime) {
        // Handle mouse input for camera control
        GLFWwindow* window = vizManager.getWindow();
        
        // Get mouse position
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        
        // Check mouse button
        int leftButton = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        
        if (leftButton == GLFW_PRESS) {
            if (leftMouseDown) {
                // Orbit camera
                float deltaX = static_cast<float>(mouseX - lastMouseX);
                float deltaY = static_cast<float>(mouseY - lastMouseY);
                camera.orbit(deltaX * 0.01f, deltaY * 0.01f);
            }
            leftMouseDown = true;
        } else {
            leftMouseDown = false;
        }
        
        lastMouseX = mouseX;
        lastMouseY = mouseY;
        
        // Keyboard controls
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            camera.pan(0.0f, 0.5f);
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            camera.pan(0.0f, -0.5f);
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            camera.pan(-0.5f, 0.0f);
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            camera.pan(0.5f, 0.0f);
        }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            camera.zoom(-0.5f);
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            camera.zoom(0.5f);
        }
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            vizManager.close();
        }
    });
    
    // Set render callback
    vizManager.setRenderCallback([&](double deltaTime) {
        // Enable depth testing
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        // Render network
        networkRenderer.render(adapter, camera, renderConfig);
        
        // ImGui control panel
        ImGui::Begin("Network Visualization Controls");
        
        ImGui::Text("Network Statistics:");
        ImGui::Text("  Neurons: %zu", adapter.getNeuronCount());
        ImGui::Text("  Synapses: %zu", adapter.getSynapseCount());
        
        ImGui::Separator();
        
        ImGui::Text("Rendering:");
        ImGui::Checkbox("Render Neurons", &renderConfig.renderNeurons);
        ImGui::Checkbox("Render Synapses", &renderConfig.renderSynapses);
        ImGui::Checkbox("Enable Lighting", &renderConfig.enableLighting);
        
        ImGui::Separator();
        
        ImGui::Text("Neuron Settings:");
        ImGui::SliderFloat("Base Radius", &renderConfig.neuronBaseRadius, 0.1f, 2.0f);
        ImGui::SliderFloat("Activity Scale", &renderConfig.neuronActivityScale, 0.0f, 2.0f);
        
        ImGui::Separator();
        
        ImGui::Text("Synapse Settings:");
        ImGui::SliderFloat("Base Thickness", &renderConfig.synapseBaseThickness, 0.01f, 0.5f);
        
        ImGui::Separator();
        
        auto stats = networkRenderer.getStats();
        ImGui::Text("Render Statistics:");
        ImGui::Text("  Neurons Rendered: %zu", stats.neuronsRendered);
        ImGui::Text("  Synapses Rendered: %zu", stats.synapsesRendered);
        ImGui::Text("  Draw Calls: %zu", stats.drawCalls);
        
        ImGui::Separator();
        
        ImGui::Text("Camera:");
        auto camPos = camera.getPosition();
        ImGui::Text("  Position: (%.1f, %.1f, %.1f)", camPos.x, camPos.y, camPos.z);
        ImGui::Text("  Distance: %.1f", camera.getDistance());
        
        ImGui::Separator();
        
        ImGui::Text("Controls:");
        ImGui::BulletText("Left Mouse: Orbit camera");
        ImGui::BulletText("WASD: Pan camera");
        ImGui::BulletText("Q/E: Zoom in/out");
        ImGui::BulletText("ESC: Exit");
        
        ImGui::End();
    });
    
    std::cout << std::endl;
    std::cout << "Starting visualization..." << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  - Left Mouse: Orbit camera" << std::endl;
    std::cout << "  - WASD: Pan camera" << std::endl;
    std::cout << "  - Q/E: Zoom in/out" << std::endl;
    std::cout << "  - ESC: Exit" << std::endl;
    std::cout << std::endl;
    
    // Run visualization
    vizManager.run();
    
    std::cout << "Visualization closed." << std::endl;
    
    return 0;
}

