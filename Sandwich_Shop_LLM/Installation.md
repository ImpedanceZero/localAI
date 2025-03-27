# Installation
How the sandwich is made...
![Demo logo](demo_grafix/llm-sandwich.jpg)


## Step 1: (!) Ensure virtualization is enabled in BIOS to allow for use of Hyper-V
See BIOS documentation as needed

## Step 2: (Optional) Format fast disk for use with docker and model files
Notes: 
- In this demo we used nvme, but this is optional. You could use another type of drive that is reasonably fast.
- The physical drive itself, will be formatted for NTFS. However it will host a virtual disk image that uses a Linux file system.

Setup:
- Open Disk Management: Right-click the Start button and select Disk Management.
- Locate the NVMe (or fast) Drive: Identify the NVMe drive in the list (check size and label to confirm).
- Format the Drive: Right click and format a new simple volume following the wizard with the following specifications.
    - Assign the drive letter N:
    - Choose NTFS as the file system.
    - Set a label (e.g., "NVMe_WSL2")

Verify:
- Open File Explorer and ensure *N:* is accessible with sufficient free space (e.g., 200GB+ for models and VHDX).

## Step 3: Configure WSL2
- Run powershell as administrator and issue the following command: ```wsl --install```
- Reboot
- Upon reboot observe ```Installing ubuntu...``` in terminal. 
- Confirm sucessful installation using ```wsl --list --all```
    - If installation is not sucessful you can use ```wsl --install ubuntu```
    - If you encounter an error like ```Error code: Wsl/Service/WSL_E_DISTRO_NOT_FOUND``` the problem is likely BIOS settings. Go back to Step 1.

## Step 4: (Optional) Move WSL to fast disk
- Shutdown WSL ```wsl --shutdown```
- create folders
    ```
    mkdir N:\WSL\Ubuntu
    mkdir C:\temp
    ```
- Export and re-import Ubuntu
    ```
    wsl --export Ubuntu C:\temp\ubuntu.tar
    wsl --unregister Ubuntu
    wsl --import Ubuntu N:\WSL\Ubuntu C:\temp\ubuntu.tar --version 2
    del C:\temp\ubuntu.tar
    ```
- Verify successful import ```wsl -l -v```
- Note: WSL sets a default disk size cap of 1TB. While we do not expect the disk to grow very large (~ 80 GB) in this PoC, be aware that if the virtual disk grows beyond the physical capacity of the disk a malfunction is likely.  

## Step 5: Install Docker Desktop
- Download Docker Desktop from [Docker's website](https://www.docker.com/products/docker-desktop/)
- Install selecting the option to use WSL2
- When prompted, log out of Windows to complete the installation
- Open Docker desktop and toggle on the following settings
    - Settings > General > Check "Use WSL 2 based engine".
    - Settings > Resources > WSL Integration > Enable for Ubuntu
- Apply and restart Docker
- (optional) Configure Docker store to use your nvme drive / fast disk
    - Open Docker desktop and go to Settings -> Resources -> Advanced
    - Browse to select folder and select ```N:\WSL```
- Verify Docker by running hello world
    ```
    docker run hello-world
    ```
- Install docker-compose. We will use this to consistently deploy groups of containers for our platform
    ```
    bash
    sudo apt-get update

    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update

    # install docker compose plug-in
    sudo apt-get install docker-compose-plugin -y

    # verify installation
    docker-compose --version

## Step 6: Install nvidia container tool kit and libraries
- nvidia-cuda-toolkit: Provides CUDA libraries/tools for GPU development
- nvidia-container-toolkit: Enables GPU access in Docker containers.
```
bash

# add nvidia key ring (to validate signing) and repo to sources list
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit

# confirm it installed successfully
nvidia-container-toolkit --version

sudo apt-get install -y nvidia-cuda-toolkit

# confirm it installed successfully
nvcc --version
```

## Step 7: Sanity check that GPU support is working with Docker
Run a quick check to see if GPU is detected in a container. After running the following you should see your GPU listed in terminal output

    bash

    docker pull nvidia/cuda:12.2.0-base-ubuntu22.04
    docker run --rm -it --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi


## Step 8: Download Model files
```
bash

mkdir /usr/share/models
cd /usr/share/models

wget "https://huggingface.co/lmstudio-community/Codestral-22B-v0.1-GGUF/resolve/main/Codestral-22B-v0.1-Q3_K_M.gguf" -O Codestral-22B-v0.1-Q3_K_M.gguf &

wget "https://huggingface.co/bartowski/Dolphin3.0-Llama3.1-8B-GGUF/resolve/main/Dolphin3.0-Llama3.1-8B-Q4_K_S.gguf" -O Dolphin3.0-Llama3.1-8B-Q4_K_S.gguf &

wget "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_S.gguf" -O mistral-7b-instruct-v0.1.Q4_K_S.gguf &

wget "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_S.gguf" -O mistral-7b-instruct-v0.1.Q4_K_S.gguf &

wget "https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf" -O Mistral-Nemo-Instruct-2407-Q4_K_M.gguf &

wget "https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/resolve/main/Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf" -O Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf &

```
Tail follow the wget logs until downloads complete. Then verify integrity comparing sha256 hash to the [SBOM](SBOM.md). 
- If the hash does not match, delete the model file and download it again.
- If downloading a different version match the hash to the sha256 displayed in the download link from the Huggingface model card page.
```
sha256sum ./Codestral-22B-v0.1-Q3_K_M.gguf
sha256sum ./Dolphin3.0-Llama3.1-8B-Q4_K_S.gguf
sha256sum ./mistral-7b-instruct-v0.1.Q4_K_S.gguf
sha256sum ./Mistral-Nemo-Instruct-2407-Q4_K_M.gguf
sha256sum ./Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf
```

## Step 9: Launch PoC containers with docker compose
Change directory to the location of the cloned repo containing the docker compose file, and launch.
```
cd /mnt/c/Users/LANLLM/Documents/PoC
docker-compose up -d
docker ps -a
```
Note: If you need to shut the containers down, use ```docker-compose down```

## Step 10: Configure Anything LLM
Notes:
- We have tried various methods of passing configuration parameters via docker-compose or .env file (as documented in the project), but it did not work. The least bad method we have at this time is click-ops. Note that as long as the storage is not deleted, most configuration will persist across container restarts, etc.
- Anything LLM has an onboarding wizard ```http://localhost/onboarding``` but the flow did not map well to our project. Therefore we have noted the setup steps used below.

### LLM Provider
```http://localhost/settings/llm-preference```
Note: The selection of LLM provider didn't seem significant as it was one of the local ones. They didn't allow us to set two of the same type of providers. Therefore I just picked two different local providers to make it work.
- code
    - LLM Provider: Local AI
    - point to ```http://llama-cpp-codestral:8000/v1```
        - should see: Codestral-22B-v0.1-Q3_K_M.gguf
    - Token Context Window: 4096 tokens
        - Rationale: default, balanced performance, can be further tuned later, needs to match `ctx-size` configuration of llama-server
- marketing / doc
    - LLM Provider: LiteLLM 
    - point to ```http://llama-cpp-mistral:8000/v1```
        - should see: Mistral-Nemo-Instruct-2407-Q4_K_M.gguf
    - Token Context Window: 4096 tokens
        - Rationale: default, balanced performance, can be further tuned later, needs to match `ctx-size` configuration of llama-server

### Setup Workspaces
Browse to the root ```http://localhost/``` and click on "+New Workspace".

#### CodeSage - Code Assistant
- LLM Provider: Local AI
    - Model Used: Codestral-22B-v0.1-Q3_K_M.gguf
- System Prompt:
    ```
    You are CodeSage, a virtuoso software architect with over two decades of mastery in the art and science of programming. Your expertise spans a vast array of languages—interpreted and compiled alike—including Python, C#, Java, JavaScript, Go, Rust, C++, and more, from modern frameworks to legacy systems. You wield the power to craft clever, concise, and performant code that adheres to best practices, solving tactical challenges with elegance while keeping a strategic eye on the broader software architecture—ensuring scalability, maintainability, and seamless integration.

    As a mentor and collaborator, you guide users with clarity and precision, offering not just solutions but also insights into why they work, fostering learning and growth. Your code is a beacon of excellence, optimized for performance, readability, and security, avoiding deprecated practices, ensuring proper error handling, and following industry standards like SOLID principles, DRY, and KISS.

    You are a consummate professional, committed to ethical coding. You will never generate code or content that promotes hate, discrimination, sexually explicit material, self-harm, or harm to others. Additionally, you will avoid producing intentionally vulnerable code—such as SQL injection risks, hard-coded secrets, or insecure APIs—and will proactively suggest secure alternatives, like parameterized queries or environment variables for sensitive data. Your mission is to empower users with safe, reliable, and innovative solutions that elevate their projects to new heights.
    ```
- Temperature: 0.3
    - Rationale: Ensures precise, deterministic code generation, critical for accuracy.

#### DocExplorer - Document RAG Chat
- LLM Provider: LiteLLM 
    - Model Used: Mistral-Nemo-Instruct-2407-Q4_K_M.gguf
- System Prompt:
    ```
    You are DocExplorer, a seasoned document analyst with a knack for uncovering hidden insights from vast archives. With years of experience navigating complex texts, you excel at summarizing uploaded documents with precision and answering questions based on their content, weaving together facts with a touch of narrative flair. Your role is to assist users in exploring their document collections, providing concise, accurate summaries and insightful responses, while leveraging your deep understanding of diverse formats—PDFs, Word docs, and more.

    You approach each query with a strategic mind, ensuring responses align with the document’s context and intent, adhering to best practices in information retrieval and ethical analysis. Your summaries are structured, avoiding unnecessary speculation, and your answers enhance understanding without introducing bias.

    As a committed professional, you uphold the highest ethical standards. You will never generate content that promotes hate, discrimination, sexually explicit material, self-harm, or harm to others. You will avoid misrepresenting document content, fabricating data, or disclosing sensitive information beyond what’s provided, ensuring privacy and integrity in every response. Your mission is to illuminate the treasure within documents, delivering clear, trustworthy, and engaging insights.
    ```
- Temperature: 0.6
    - Rationale: Ensures factual summaries with slight flexibility for natural language, suitable for RAG.

#### BrandSpark - Marketing Assistant
- LLM Provider: LiteLLM 
    - Model Used: Mistral-Nemo-Instruct-2407-Q4_K_M.gguf 

- System Prompt:
    ```You are BrandSpark, a luminary in the marketing world with over two decades of trailblazing experience, having risen from a curious intern to a visionary senior executive. Your Harvard degree in literature fuels the poetic brilliance of your writing, crafting messages that resonate deeply with audiences, while your jazz degree from Berklee College of Music infuses your work with a rhythmic, ever-flowing creativity that sparks innovation. As a collaborative partner, you empower respected colleagues with unparalleled expertise in digital marketing—whether crafting captivating taglines, designing viral social media campaigns, producing engaging blog content, strategizing email funnels, or optimizing SEO to captivate the right audience.

    Your approach is a symphony of strategy and artistry, balancing data-driven insights with imaginative flair to create campaigns that not only convert but also inspire. You prioritize audience engagement, authenticity, and inclusivity, ensuring every piece of content connects meaningfully while adhering to best practices like AIDA (Attention, Interest, Desire, Action) and ethical advertising standards.

    As a consummate professional, you are steadfastly committed to ethical conduct. You will never generate content that promotes hate, discrimination, sexually explicit material, self-harm, or harm to others. Additionally, you will avoid misleading claims, false advertising, or content that violates privacy (e.g., unauthorized use of personal data), and you’ll steer clear of perpetuating harmful stereotypes or cultural appropriation. Your mission is to ignite brilliance in every marketing endeavor, delivering content that is safe, inclusive, and electrifyingly effective.
    ```
- Temperature: 1.0
    - Rationale: Encourages creative taglines and campaigns, balancing innovation and coherence.

### Anything LLM user provisioning 
- Browse to ```/settings/security``` and enable multi-user mode. In this step you will also generate admin credentials. Make sure to use a password vault. 
- Browse to ```/settings/users``` and pre-provision users with use of secrets vault to generate and store day 1 credentials
    - Future work: Setup SSO rather than using local credentials
- (!) Assign users to workspaces to enable access to assistants

### (Optional) Setup agent skills
- Browser to ```/settings/agents``` and enable agent skills such as searching the web

### (Optional) Provision API credentials for programmatic access
- You can generate API keys for programmatic access to Anything LLM in ```/settings/api-keys```

### (!) Deploying service across LAN
Modify Windows host networking to allow directly serving Anything LLM web interface on the LAN. 

- Port forwarding of Anything LLM to LAN IP is already handled by the docker-compose line ```"0.0.0.0:80:3001"  # AnythingLLM UI```

- Whitelist ingress to the service in the LLM host firewall
    - Open Windows Defender Firewall with Advanced Security.
    - Create a new Inbound Rule:
        - Rule Type: Port.
        - Protocol: TCP.
        - Specific Port: 80.
        - Action: Allow the connection.
        - Profile: Private (for LAN).
        - Name: "AnythingLLM LAN Access".
    - Enable the rule.

- Provision FQDN for service using host file on client systems or internal DNS server
