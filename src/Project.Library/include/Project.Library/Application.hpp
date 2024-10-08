#pragma once
#include <cstdint>
#include <glad/glad.h>

struct GLFWwindow;

class Application
{
public:
    void Run();
    GLuint _quadVAO = 0;
    GLuint _quadVBO = 0;

protected:
    void Close();
    bool IsKeyPressed(int32_t key);
    
    double GetDeltaTime();

    virtual void AfterCreatedUiContext();
    virtual void BeforeDestroyUiContext();
    virtual bool Initialize();
    virtual bool Load();
    virtual void Unload();
    virtual void RenderScene(float deltaTime);
    virtual void RenderUI(float deltaTime);
    virtual void Update(float deltaTime);


private:
    GLFWwindow* _windowHandle = nullptr;
    void Render(float deltaTime);

};