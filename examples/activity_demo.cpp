#include "snnfw/Datastore.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/NetworkInspector.h"
#include "snnfw/ActivityMonitor.h"
#include "snnfw/NetworkDataAdapter.h"
#include "snnfw/LayoutEngine.h"
#include "snnfw/NetworkGraphRenderer.h"
#include "snnfw/ActivityVisualizer.h"
#include "snnfw/SpikeRenderer.h"
#include "snnfw/RecordingManager.h"
#include "snnfw/RasterPlotRenderer.h"
#include "snnfw/InteractionManager.h"
#include "snnfw/PatternDetector.h"
#include "snnfw/ActivityHistogram.h"
#include "snnfw/VisualizationManager.h"
#include "snnfw/Camera.h"
#include "snnfw/ShaderManager.h"
#include <imgui.h>
#include <iostream>
#include <random>
#include <chrono>

using namespace snnfw;

int main() {
    std::cout << "SNNFW Activity Visualization Demo" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Initialize datastore
    Datastore datastore("./activity_viz_db", 100000);
    
    // Create components
    NeuralObjectFactory factory;
    NetworkInspector inspector;
    ActivityMonitor activityMonitor(datastore);
    NetworkDataAdapter adapter(datastore, inspector, &activityMonitor);
    
    // Build a simple test network with 50 neurons
    std::cout << "Building test network..." << std::endl;
    
    auto cluster = factory.createCluster();
    
    // Create 50 neurons
    std::vector<uint64_t> neuronIds;
    for (int i = 0; i < 50; ++i) {
        auto neuron = factory.createNeuron(50.0, 0.95, 20);
        
        // Create axon and dendrite
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
    
    // Create some synapses between neurons
    std::cout << "Creating synapses..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> neuronDist(0, neuronIds.size() - 1);
    std::uniform_real_distribution<> weightDist(0.5, 2.0);
    
    std::vector<uint64_t> synapseIds;
    for (int i = 0; i < 100; ++i) {
        int sourceIdx = neuronDist(gen);
        int targetIdx = neuronDist(gen);
        
        if (sourceIdx == targetIdx) continue;
        
        auto sourceNeuron = datastore.getNeuron(neuronIds[sourceIdx]);
        auto targetNeuron = datastore.getNeuron(neuronIds[targetIdx]);
        
        if (!sourceNeuron || !targetNeuron) continue;
        
        auto axon = datastore.getAxon(sourceNeuron->getAxonId());
        auto dendriteIds = targetNeuron->getDendriteIds();
        
        if (!axon || dendriteIds.empty()) continue;
        
        auto dendrite = datastore.getDendrite(dendriteIds[0]);
        if (!dendrite) continue;
        
        // Create synapse
        double weight = weightDist(gen);
        auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), weight, 1.0);

        // Add synapse to axon
        axon->addSynapse(synapse->getId());

        // Store synapse and updated axon
        datastore.put(synapse);
        datastore.put(axon);
        synapseIds.push_back(synapse->getId());
    }
    
    std::cout << "Created cluster with " << neuronIds.size() << " neurons and "
              << synapseIds.size() << " synapses" << std::endl;

    // Initialize visualization manager
    VisualizationManager vizManager(800, 600, "SNNFW Activity Visualization");
    vizManager.setTargetFPS(60);
    vizManager.enableVSync(true);
    vizManager.setBackgroundColor(0.1f, 0.1f, 0.15f);

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    
    // Extract network
    std::cout << "Extracting network structure..." << std::endl;
    if (!adapter.extractHierarchy(cluster->getId(), "Cluster")) {
        std::cerr << "Failed to extract network!" << std::endl;
        return 1;
    }
    
    std::cout << "Extracted " << adapter.getNeurons().size() << " neurons" << std::endl;
    std::cout << "Extracted " << adapter.getSynapses().size() << " synapses" << std::endl;
    
    // Compute layout
    std::cout << "Computing layout..." << std::endl;
    LayoutEngine layoutEngine;
    LayoutConfig layoutConfig;
    layoutConfig.algorithm = LayoutAlgorithm::CIRCULAR;
    layoutConfig.neuronSpacing = 3.0f;
    
    if (!layoutEngine.computeLayout(adapter, layoutConfig)) {
        std::cerr << "Failed to compute layout!" << std::endl;
        return 1;
    }
    
    adapter.updateSynapsePositions();
    std::cout << "Layout computed successfully!" << std::endl;
    
    // Create renderers
    ShaderManager shaderManager;
    NetworkGraphRenderer networkRenderer(shaderManager);
    SpikeRenderer spikeRenderer(shaderManager);
    ActivityVisualizer activityVisualizer(activityMonitor, adapter);
    RecordingManager recordingManager(activityVisualizer);
    RasterPlotRenderer rasterPlotRenderer(shaderManager);
    InteractionManager interactionManager(adapter);
    PatternDetector patternDetector;
    ActivityHistogram activityHistogram(shaderManager);

    if (!networkRenderer.initialize()) {
        std::cerr << "Failed to initialize network renderer!" << std::endl;
        return 1;
    }

    if (!spikeRenderer.initialize()) {
        std::cerr << "Failed to initialize spike renderer!" << std::endl;
        return 1;
    }

    if (!rasterPlotRenderer.initialize()) {
        std::cerr << "Failed to initialize raster plot renderer!" << std::endl;
        return 1;
    }

    activityHistogram.initialize();
    
    std::cout << "Renderers initialized!" << std::endl;

    // Setup raster plot neuron mapping
    rasterPlotRenderer.setNeuronMapping(neuronIds);

    // Create camera
    Camera camera;
    camera.setPosition(glm::vec3(0.0f, 0.0f, 30.0f));
    camera.lookAt(glm::vec3(0.0f, 0.0f, 0.0f));
    
    // Activity configuration
    ActivityConfig activityConfig;
    activityConfig.showPropagation = true;
    activityConfig.showHeatmap = true;
    activityConfig.decayRate = 1.5f;
    activityConfig.particleLifetime = 800;
    activityConfig.propagationSpeed = 10.0f;
    activityVisualizer.setConfig(activityConfig);

    // Raster plot configuration
    RasterPlotConfig rasterConfig;
    rasterConfig.timeWindowMs = 2000.0f;
    rasterConfig.spikeMarkerSize = 4.0f;
    rasterConfig.colorByNeuronType = true;
    rasterConfig.showGrid = true;
    rasterPlotRenderer.setConfig(rasterConfig);
    
    // Render configuration
    RenderConfig renderConfig;
    renderConfig.mode = RenderMode::NEURONS_AND_SYNAPSES;
    renderConfig.enableLighting = true;
    
    SpikeRenderConfig spikeConfig;
    spikeConfig.showTrails = true;
    spikeConfig.glowIntensity = 2.0f;
    
    std::cout << "\nStarting visualization..." << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  - Left Mouse: Orbit camera" << std::endl;
    std::cout << "  - WASD: Pan camera" << std::endl;
    std::cout << "  - Q/E: Zoom in/out" << std::endl;
    std::cout << "  - ESC: Exit" << std::endl;
    std::cout << "\nGenerating random spike activity..." << std::endl;
    
    // Timing
    auto startTime = std::chrono::high_resolution_clock::now();
    uint64_t simulationTime = 0;

    // Random spike generation
    std::uniform_int_distribution<> spikeDist(0, 99);
    uint64_t lastSpikeTime = 0;
    uint64_t spikeInterval = 50; // Generate spike every 50ms

    // Mouse interaction state
    bool mousePressed = false;
    double mouseX = 0.0, mouseY = 0.0;
    int screenWidth = 1280, screenHeight = 720;

    // Set up mouse click callback
    vizManager.setMouseClickCallback([&](int button, int action, int mods, double x, double y) {
        // Update mouse position
        mouseX = x;
        mouseY = y;

        // Handle left click for selection
        if (button == 0 && action == 1) {  // GLFW_MOUSE_BUTTON_LEFT and GLFW_PRESS
            PickResult result = interactionManager.pickNeuron(
                static_cast<float>(x), static_cast<float>(y),
                screenWidth, screenHeight, camera);

            if (result.hit) {
                // Determine selection mode based on modifiers
                SelectionMode mode = SelectionMode::SINGLE;
                if (mods & 0x0002) {  // GLFW_MOD_CONTROL
                    mode = SelectionMode::ADDITIVE;
                } else if (mods & 0x0004) {  // GLFW_MOD_ALT
                    mode = SelectionMode::SUBTRACTIVE;
                } else if (mods & 0x0001) {  // GLFW_MOD_SHIFT
                    mode = SelectionMode::TOGGLE;
                }

                interactionManager.selectNeuron(result.neuronId, mode);
            } else {
                // Clicked on empty space - clear selection if no modifiers
                if (mods == 0) {
                    interactionManager.clearSelection();
                }
            }
        }
    });

    // Set up update callback (called every frame before rendering)
    vizManager.setUpdateCallback([&](double deltaTime) {
        // Get current window size
        glfwGetFramebufferSize(vizManager.getWindow(), &screenWidth, &screenHeight);

        // Get current mouse position
        glfwGetCursorPos(vizManager.getWindow(), &mouseX, &mouseY);
        // Calculate time
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - startTime).count();
        simulationTime = static_cast<uint64_t>(elapsed);

        // Generate random spikes
        if (simulationTime - lastSpikeTime >= spikeInterval && !synapseIds.empty()) {
            int synapseIdx = spikeDist(gen) % synapseIds.size();
            auto synapse = datastore.getSynapse(synapseIds[synapseIdx]);

            if (synapse) {
                auto axon = datastore.getAxon(synapse->getAxonId());
                auto dendrite = datastore.getDendrite(synapse->getDendriteId());

                if (axon && dendrite) {
                    uint64_t sourceNeuronId = axon->getSourceNeuronId();
                    uint64_t targetNeuronId = dendrite->getTargetNeuronId();

                    // Record spike in activity visualizer
                    activityVisualizer.recordSpike(sourceNeuronId, targetNeuronId,
                                                  synapse->getId(), simulationTime);

                    // Record in raster plot
                    rasterPlotRenderer.addSpike(sourceNeuronId, simulationTime, true);

                    // Record in recording manager if recording
                    if (recordingManager.isRecording()) {
                        RecordedSpike recordedSpike;
                        recordedSpike.timestamp = simulationTime;
                        recordedSpike.sourceNeuronId = sourceNeuronId;
                        recordedSpike.targetNeuronId = targetNeuronId;
                        recordedSpike.synapseId = synapse->getId();
                        recordingManager.recordSpike(recordedSpike);
                    }

                    // Record in pattern detector
                    const auto& neurons = adapter.getNeurons();
                    for (const auto& neuron : neurons) {
                        if (neuron.id == sourceNeuronId) {
                            glm::vec3 pos(neuron.position.x, neuron.position.y, neuron.position.z);
                            patternDetector.recordSpike(sourceNeuronId, simulationTime, pos);
                            break;
                        }
                    }

                    // Also record in activity monitor for statistics
                    activityMonitor.recordNeuronFiring(sourceNeuronId,
                                                      static_cast<double>(simulationTime));
                    activityMonitor.recordNeuronFiring(targetNeuronId,
                                                      static_cast<double>(simulationTime));
                }
            }

            lastSpikeTime = simulationTime;
        }

        // Update activity visualizer
        activityVisualizer.update(simulationTime);

        // Update recording manager (for playback)
        if (deltaTime > 0.0) {
            recordingManager.update(static_cast<uint64_t>(deltaTime * 1000.0));
        }

        // Update pattern detector
        patternDetector.update(simulationTime);

        // Update activity histogram
        activityHistogram.update(activityVisualizer, adapter, simulationTime);

        // Apply activity heatmap to neurons
        if (activityConfig.showHeatmap) {
            spikeRenderer.applyActivityHeatmap(
                activityVisualizer.getNeuronActivity(), adapter);
        }

        // Update hover state (mouse picking for hover effect)
        PickResult hoverResult = interactionManager.pickNeuron(
            static_cast<float>(mouseX), static_cast<float>(mouseY),
            screenWidth, screenHeight, camera);

        if (hoverResult.hit) {
            interactionManager.setHoveredNeuron(hoverResult.neuronId);
        } else {
            interactionManager.clearHover();
        }
    });

    // Set up render callback
    vizManager.setRenderCallback([&](double deltaTime) {

        // Render network
        networkRenderer.render(adapter, camera, renderConfig);

        // Render selection highlighting
        if (interactionManager.getSelectionCount() > 0) {
            networkRenderer.renderSelectedNeurons(
                interactionManager.getSelectedNeurons(),
                adapter, camera, interactionManager.getHighlightColor());
        }

        // Render hover highlighting
        if (interactionManager.getHoveredNeuron() != 0) {
            std::unordered_set<uint64_t> hovered = {interactionManager.getHoveredNeuron()};
            networkRenderer.renderSelectedNeurons(
                hovered, adapter, camera, glm::vec4(0.5f, 0.8f, 1.0f, 1.0f));
        }

        // Render spike particles and trails
        spikeRenderer.renderSpikeParticles(
            activityVisualizer.getSpikeParticles(), camera, spikeConfig);
        spikeRenderer.renderSpikeTrails(
            activityVisualizer.getSpikeParticles(), adapter, camera, spikeConfig);

        // Render UI
        ImGui::Begin("Activity Visualization");
        ImGui::Text("Simulation Time: %lu ms", static_cast<unsigned long>(simulationTime));
        ImGui::Text("Total Spikes: %u", activityVisualizer.getTotalSpikes());
        ImGui::Text("Active Neurons: %u", activityVisualizer.getActiveNeuronCount());
        ImGui::Text("Avg Activity: %.3f", activityVisualizer.getAverageActivityLevel());
        ImGui::Text("Active Particles: %zu", activityVisualizer.getSpikeParticles().size());

        ImGui::Separator();
        ImGui::Text("Visualization");
        ImGui::Checkbox("Show Propagation", &activityConfig.showPropagation);
        ImGui::Checkbox("Show Heatmap", &activityConfig.showHeatmap);
        ImGui::Checkbox("Show Trails", &spikeConfig.showTrails);
        ImGui::SliderFloat("Decay Rate", &activityConfig.decayRate, 0.1f, 5.0f);
        ImGui::SliderFloat("Glow Intensity", &spikeConfig.glowIntensity, 0.5f, 5.0f);
        ImGui::SliderInt("Spike Interval (ms)", reinterpret_cast<int*>(&spikeInterval), 10, 500);

        activityVisualizer.setConfig(activityConfig);

        ImGui::Separator();
        ImGui::Text("Recording");

        if (recordingManager.isRecording()) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "RECORDING");
            if (ImGui::Button("Stop Recording")) {
                recordingManager.stopRecording();
            }
        } else {
            if (ImGui::Button("Start Recording")) {
                recordingManager.startRecording();
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Save")) {
            recordingManager.saveRecording("recording.snnr");
        }

        ImGui::SameLine();
        if (ImGui::Button("Load")) {
            recordingManager.loadRecording("recording.snnr");
        }

        const auto& playbackState = recordingManager.getPlaybackState();
        const auto& metadata = recordingManager.getMetadata();

        ImGui::Text("Recorded Spikes: %zu", metadata.spikeCount);

        if (playbackState.playing) {
            if (ImGui::Button("Pause")) {
                recordingManager.pause();
            }
        } else {
            if (ImGui::Button("Play")) {
                recordingManager.play();
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Stop")) {
            recordingManager.stop();
        }

        float speed = playbackState.speed;
        if (ImGui::SliderFloat("Speed", &speed, 0.1f, 5.0f)) {
            recordingManager.setSpeed(speed);
        }

        bool looping = playbackState.looping;
        if (ImGui::Checkbox("Loop", &looping)) {
            recordingManager.setLooping(looping);
        }

        ImGui::End();

        // Raster plot window
        bool showRasterPlot = true;
        if (showRasterPlot) {
            ImGui::Begin("Raster Plot", &showRasterPlot);

            // Get window size
            ImVec2 windowSize = ImGui::GetContentRegionAvail();

            // Reserve space for controls
            float controlHeight = 120.0f;
            float plotHeight = windowSize.y - controlHeight;

            if (plotHeight > 100.0f) {
                // Render raster plot
                ImVec2 plotPos = ImGui::GetCursorScreenPos();
                rasterPlotRenderer.render(plotPos.x, plotPos.y, windowSize.x, plotHeight,
                                         simulationTime);

                // Advance cursor
                ImGui::Dummy(ImVec2(windowSize.x, plotHeight));
            }

            ImGui::Separator();
            ImGui::Text("Raster Plot Controls");
            ImGui::Text("Total Spikes in Plot: %zu", rasterPlotRenderer.getSpikeCount());

            if (ImGui::SliderFloat("Time Window (ms)", &rasterConfig.timeWindowMs, 500.0f, 5000.0f)) {
                rasterPlotRenderer.setConfig(rasterConfig);
            }

            if (ImGui::SliderFloat("Marker Size", &rasterConfig.spikeMarkerSize, 1.0f, 10.0f)) {
                rasterPlotRenderer.setConfig(rasterConfig);
            }

            if (ImGui::Checkbox("Show Grid", &rasterConfig.showGrid)) {
                rasterPlotRenderer.setConfig(rasterConfig);
            }

            ImGui::SameLine();
            if (ImGui::Checkbox("Color by Type", &rasterConfig.colorByNeuronType)) {
                rasterPlotRenderer.setConfig(rasterConfig);
            }

            if (ImGui::Button("Clear Raster Plot")) {
                rasterPlotRenderer.clearSpikes();
            }

            ImGui::End();
        }

        // Interaction window
        bool showInteraction = true;
        if (showInteraction) {
            ImGui::Begin("Interaction", &showInteraction);

            // Selection info
            ImGui::Text("Selection");
            ImGui::Text("Selected Neurons: %zu", interactionManager.getSelectionCount());

            if (interactionManager.getSelectionCount() > 0) {
                auto stats = interactionManager.getSelectionStats();
                ImGui::Text("Excitatory: %zu", stats.excitatoryCount);
                ImGui::Text("Inhibitory: %zu", stats.inhibitoryCount);
                ImGui::Text("Avg Activity: %.3f", stats.avgActivity);
                ImGui::Text("Avg Connections: %d", stats.avgConnections);

                if (ImGui::Button("Clear Selection")) {
                    interactionManager.clearSelection();
                }

                ImGui::SameLine();
                if (ImGui::Button("Focus on Selection")) {
                    interactionManager.focusOnSelection(camera, 2.0f);
                }
            }

            ImGui::Separator();
            ImGui::Text("Selection Tools");

            if (ImGui::Button("Select All")) {
                interactionManager.selectAll();
            }

            ImGui::SameLine();
            if (ImGui::Button("Invert Selection")) {
                interactionManager.invertSelection();
            }

            // Select by activity
            static float minActivity = 0.0f;
            static float maxActivity = 1.0f;
            ImGui::SliderFloat("Min Activity", &minActivity, 0.0f, 1.0f);
            ImGui::SliderFloat("Max Activity", &maxActivity, 0.0f, 1.0f);
            if (ImGui::Button("Select by Activity")) {
                interactionManager.selectByActivity(minActivity, maxActivity);
            }

            // Select by connectivity
            static int minConnections = 0;
            static int maxConnections = 100;
            ImGui::SliderInt("Min Connections", &minConnections, 0, 100);
            ImGui::SliderInt("Max Connections", &maxConnections, 0, 100);
            if (ImGui::Button("Select by Connectivity")) {
                interactionManager.selectByConnectivity(minConnections, maxConnections);
            }

            ImGui::Separator();
            ImGui::Text("Camera Bookmarks");

            static char bookmarkName[64] = "Bookmark1";
            ImGui::InputText("Name", bookmarkName, sizeof(bookmarkName));

            if (ImGui::Button("Add Bookmark")) {
                interactionManager.addBookmark(std::string(bookmarkName), camera);
            }

            const auto& bookmarks = interactionManager.getBookmarks();
            for (const auto& bookmark : bookmarks) {
                if (ImGui::Button(bookmark.name.c_str())) {
                    interactionManager.applyBookmark(bookmark.name, camera);
                }
                ImGui::SameLine();
                std::string removeLabel = "X##" + bookmark.name;
                if (ImGui::SmallButton(removeLabel.c_str())) {
                    interactionManager.removeBookmark(bookmark.name);
                }
            }

            ImGui::Separator();
            ImGui::Text("Hover Info");
            if (interactionManager.getHoveredNeuron() != 0) {
                ImGui::Text("Hovered Neuron ID: %lu",
                           static_cast<unsigned long>(interactionManager.getHoveredNeuron()));
            } else {
                ImGui::Text("No neuron hovered");
            }

            ImGui::End();
        }

        // Pattern Detection window
        bool showPatterns = true;
        if (showPatterns) {
            ImGui::Begin("Pattern Detection", &showPatterns);

            ImGui::Text("Detected Patterns");
            ImGui::Separator();

            // Synchrony events
            ImGui::Text("Synchrony Events: %zu", patternDetector.getSynchronyCount());
            if (ImGui::TreeNode("Synchrony")) {
                const auto& syncEvents = patternDetector.getSynchronyEvents();
                for (size_t i = 0; i < std::min(syncEvents.size(), size_t(10)); ++i) {
                    const auto& event = syncEvents[syncEvents.size() - 1 - i];
                    ImGui::Text("  Time: %lu, Neurons: %zu, Coherence: %.2f",
                               static_cast<unsigned long>(event.timestamp),
                               event.neuronIds.size(),
                               event.coherence);
                }
                if (syncEvents.size() > 10) {
                    ImGui::Text("  ... and %zu more", syncEvents.size() - 10);
                }
                ImGui::TreePop();
            }

            // Oscillations
            ImGui::Text("Oscillations: %zu", patternDetector.getOscillationCount());
            if (ImGui::TreeNode("Oscillations")) {
                const auto& oscillations = patternDetector.getOscillations();
                for (size_t i = 0; i < std::min(oscillations.size(), size_t(10)); ++i) {
                    const auto& osc = oscillations[oscillations.size() - 1 - i];
                    ImGui::Text("  Freq: %.1f Hz, Confidence: %.2f",
                               osc.frequency, osc.confidence);
                }
                if (oscillations.size() > 10) {
                    ImGui::Text("  ... and %zu more", oscillations.size() - 10);
                }
                ImGui::TreePop();
            }

            // Waves
            ImGui::Text("Wave Events: %zu", patternDetector.getWaveCount());
            if (ImGui::TreeNode("Waves")) {
                const auto& waves = patternDetector.getWaveEvents();
                for (size_t i = 0; i < std::min(waves.size(), size_t(10)); ++i) {
                    const auto& wave = waves[waves.size() - 1 - i];
                    ImGui::Text("  Speed: %.2f, Neurons: %zu, Coherence: %.2f",
                               wave.speed, wave.neuronIds.size(), wave.coherence);
                }
                if (waves.size() > 10) {
                    ImGui::Text("  ... and %zu more", waves.size() - 10);
                }
                ImGui::TreePop();
            }

            // Bursts
            ImGui::Text("Burst Events: %zu", patternDetector.getBurstCount());
            if (ImGui::TreeNode("Bursts")) {
                const auto& bursts = patternDetector.getBurstEvents();
                for (size_t i = 0; i < std::min(bursts.size(), size_t(10)); ++i) {
                    const auto& burst = bursts[bursts.size() - 1 - i];
                    ImGui::Text("  Neuron: %lu, Spikes: %d, Freq: %.1f Hz",
                               static_cast<unsigned long>(burst.neuronId),
                               burst.spikeCount, burst.frequency);
                }
                if (bursts.size() > 10) {
                    ImGui::Text("  ... and %zu more", bursts.size() - 10);
                }
                ImGui::TreePop();
            }

            ImGui::Separator();
            if (ImGui::Button("Clear Patterns")) {
                patternDetector.clearPatterns();
            }

            ImGui::End();
        }

        // Activity Histogram window
        bool showHistogram = true;
        if (showHistogram) {
            ImGui::Begin("Activity Histogram", &showHistogram);

            ImGui::Text("Activity Distribution");
            ImGui::Separator();

            // Histogram type selection
            static int histogramType = 0;
            const char* histogramTypes[] = { "Firing Rate (Hz)", "Spike Count", "ISI (ms)", "Activity Level" };
            if (ImGui::Combo("Histogram Type", &histogramType, histogramTypes, 4)) {
                activityHistogram.setHistogramType(static_cast<HistogramType>(histogramType));
            }

            // Configuration
            auto histConfig = activityHistogram.getConfig();
            if (ImGui::SliderInt("Number of Bins", &histConfig.numBins, 5, 50)) {
                activityHistogram.setConfig(histConfig);
            }
            if (ImGui::SliderFloat("Time Window (ms)", &histConfig.timeWindowMs, 100.0f, 5000.0f)) {
                activityHistogram.setConfig(histConfig);
            }
            ImGui::Checkbox("Auto Scale", &histConfig.autoScale);
            if (!histConfig.autoScale) {
                ImGui::SliderFloat("Min Frequency", &histConfig.minFrequency, 0.0f, 50.0f);
                ImGui::SliderFloat("Max Frequency", &histConfig.maxFrequency, 10.0f, 200.0f);
            }
            ImGui::Checkbox("Show Grid", &histConfig.showGrid);
            ImGui::Checkbox("Show Mean", &histConfig.showMean);
            ImGui::Checkbox("Show Median", &histConfig.showMedian);

            ImGui::Separator();
            ImGui::Text("Statistics");

            const auto& stats = activityHistogram.getStats();
            ImGui::Text("Mean: %.2f", stats.mean);
            ImGui::Text("Median: %.2f", stats.median);
            ImGui::Text("Std Dev: %.2f", stats.stddev);
            ImGui::Text("Min: %.2f", stats.min);
            ImGui::Text("Max: %.2f", stats.max);
            ImGui::Text("Count: %d", stats.totalCount);

            ImGui::Separator();
            if (ImGui::Button("Clear Histogram")) {
                activityHistogram.clear();
            }

            ImGui::End();
        }
    });

    // Run visualization
    vizManager.run();

    // Cleanup
    networkRenderer.cleanup();
    spikeRenderer.cleanup();
    rasterPlotRenderer.cleanup();
    activityHistogram.cleanup();
    
    std::cout << "\nVisualization ended." << std::endl;
    std::cout << "Total spikes generated: " << activityVisualizer.getTotalSpikes() << std::endl;
    
    return 0;
}

