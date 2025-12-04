# Yahboom Obstacle Avoidance Task

This directory is part of the repository  
`IsaacLab-Final_Project/source/isaaclab_tasks/isaaclab_tasks/direct/yahboom`  
and contains the Yahboom obstacle avoidance task built on NVIDIA Isaac Lab.

The instructions below will help you set up the Conda environment, install dependencies (including Isaac Sim), and run the preconfigured VS Code task.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Isaacwilliam4/IsaacLab-Final_Project.git
cd IsaacLab-Final_Project
````

---

## 2. Create the Conda Environment

Use the provided helper script to create the Conda environment:

```bash
./isaaclab.sh -c isaaclab_final_project
```

This will create a Conda environment named `isaaclab_final_project`.

---

## 3. Activate the Environment and Install Dependencies

Activate the environment:

```bash
conda activate isaaclab_final_project
```

Then install the project dependencies:

```bash
./isaaclab.sh -i
```

---

## 4. Install Isaac Sim into the Environment

Install Isaac Sim (5.1.0) into the active Conda environment:

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

> **Note:** Make sure `pip` is the one from the `isaaclab_final_project` environment (it will be if the Conda env is activated).

---

## 5. Configure VS Code to Use the Environment

If you are using VS Code, select the `isaaclab_final_project` environment as the Python interpreter:

1. Press `Ctrl + Shift + P`
2. Search for **“Python: Select Interpreter”**
3. Choose the interpreter corresponding to `isaaclab_final_project`

---

## 6. Run the Yahboom Task from VS Code

A VS Code task is provided to run the Yahboom obstacle avoidance scenario.

1. Press `Ctrl + Shift + P`
2. Search for **“Run Task”**
3. Select **`play yahboom final velocity`**

If the environment and dependencies are set up correctly, the task should launch and run without additional configuration.
