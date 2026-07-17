# CHAMP
## Central High Performance Computing Facility
### CSIR–National Environmental Engineering Research Institute (CSIR-NEERI)

## 1. Introduction

CHAMP (Central High Performance Computing Facility) is the premier High Performance Computing (HPC) cluster used at CSIR–National Environmental Engineering Research Institute (CSIR-NEERI) for executing computationally intensive scientific and engineering applications. Unlike a conventional desktop or laptop, CHAMP consists of hundreds of interconnected compute nodes capable of executing large-scale computations in parallel.

The facility enables researchers and engineers to process massive datasets, perform scientific simulations, train machine learning models, and execute parallel applications that would otherwise require impractical amounts of time or memory on a personal computer. CHAMP follows a centralized architecture where users prepare and submit computational jobs to a scheduler, which automatically allocates suitable compute resources.

## 2. Objectives of CHAMP

The primary objective is to provide a centralized computing infrastructure capable of handling computational workloads beyond the capacity of standard workstations. Key objectives include:

- **Scientific Computing:** Executing large scientific computations and numerical simulations.
- **Data Processing:** Processing massive datasets efficiently for environmental analysis.
- **Advanced Modelling:** Supporting weather, climate modelling, and image processing.
- **AI & ML:** Providing specialized hardware for Artificial Intelligence and Machine Learning workloads.
- **Resource Optimization:** Reducing execution time by distributing computations across multiple nodes and ensuring efficient utilization of shared computational resources.

## 3. Why High Performance Computing?

Many scientific applications involve large datasets, high memory requirements, and intensive algorithms. Executing these on a personal computer often results in:

- **Excessive execution time:** Tasks taking days instead of hours.
- **System slowdown:** Complete consumption of local UI resources.
- **Memory exhaustion:** "Out of Memory" errors when loading complete datasets.

HPC overcomes these limitations through Parallel Computing. Instead of executing an application on a single processor, work is divided across multiple compute nodes.

```
[ SYSTEM PERFORMANCE CONTRAST ]

Single Computer (Sequential)          CHAMP Cluster (Parallel)

Entire Application                    Node 1 │ Node 2 │ Node 3 │ Node N
      │                                  ↓       ↓       ↓       ↓
20 Hours Execution Time               Distributed Task Execution
      ▼                                       │
  Final Output                        Results Combined
                                              │
                                       2 Hours Execution Time
                                              ▼
                                        Final Output
```

## 4. CHAMP System Architecture

The overall architecture follows a multi-tiered entry and execution system to ensure security and efficient load balancing.

```
[ ARCHITECTURAL LAYERS ]

 User Workstation
      │
      ▼
 Gateway Server
 (Secure Entry)
      │
      ▼
 CHAMP Login Server
 (Management & Setup)
      │
      ▼
 PBS Job Scheduler
 (Resource Manager)
      │
      ▼
 Compute Nodes (CPU/GPU)
 (Actual Computation)
      │
      ▼
 Application Output
```

### 4.1 Component Roles

- **Gateway Server:** The secure entry point acting as the first stage of authentication. It is not for computation or storage.
- **CHAMP Login Server:** The developer environment for file management, editing scripts, creating Python virtual environments, and compiling applications.
- **PBS Job Scheduler:** Manages job queues, allocates resources (CPU/GPU), and monitors execution to ensure fair sharing among users.
- **Compute Nodes:** The physical machines that perform the heavy lifting. Users do not interact with these directly.

## 5. Operational Workflow & Job Lifecycle

Based on internship training, the actual process of utilizing CHAMP follows a strict operational sequence to ensure environment stability.

### 5.1 The Logical Login Sequence

```
NEERI Network / Local Workstation
      │
 SSH to Gateway Server
      │
 SSH to CHAMP Login Node
      │
 Configure Proxy (for External Access)
      │
 Move to /scratch/<username>/
      │
 Load Environment Modules (module load)
      │
 Activate Python Virtual Environment
      │
 Submit PBS Script (qsub job.pbs)
```

### 5.2 The PBS Job Lifecycle

When a job is submitted via qsub, it undergoes the following transitions:

```
[ PBS EXECUTION STEPS ]

 User Submits PBS Job Script
      │
 Checks Resource Requirements
 (Nodes, CPUs, GPUs, Memory, Time)
      │
 Selects Queue (CPU or GPU Queue)
      │
 Job Enters "Queued" State
      │
 Scheduler Allocates Compute Nodes
      │
 Job Enters "Running" State
      │
 Resources Released on Completion
      │
 Review .o (Output) and .e (Error) Files
```

### ⚠️ CRITICAL POLICY: LOGIN NODE VS. COMPUTE NODE

| ✔ ALLOWED ON LOGIN NODE | ✗ PROHIBITED ON LOGIN NODE |
|---|---|
| • Editing Python/PBS scripts | • Running long-running simulations |
| • Compiling C++/Fortran code | • Training Deep Learning models |
| • Installing Python libraries (via proxy) | • Intensive data processing/extraction |
| • Managing directory structures | • Large-scale parallel execution |
| • Submitting and monitoring jobs | |

## 6. Storage Structure

CHAMP provides different storage locations tailored for specific purposes.

| Storage Path | Characteristics & Best Practices |
|---|---|
| **Home Directory**<br>`/home/<user>` | Personal directory (~100 GB limit). Best for small configuration files, SSH keys, and script sources. |
| **Scratch Directory**<br>`/scratch/<user>` | High-capacity storage. **Mandatory** for large datasets, cloning repositories, and the actual execution of computational jobs. |

## 7. Software & Python Environment

The cluster uses an Environment Modules system to manage conflicting software versions. The recommended workflow for Python-based research is:

1. **Configure Proxy:** Set organization proxy settings for the session to enable outbound internet.
2. **Load Module:** Use `module load python/<version>`.
3. **Isolate:** Create a dedicated Python Virtual Environment.
4. **Install:** Install specific version-controlled libraries inside that environment.

**Internship Workflow Example:**

```
export proxy... → module purge → module load python/3.10 → source venv/bin/activate → qsub script.pbs
```

## 8. Hardware Specifications

CHAMP provides specialized hardware divided into two primary clusters.

| Cluster Type | Hardware Configuration | Primary Applications |
|---|---|---|
| **CPU Cluster**<br>(410 Nodes) | RHEL 8.7, 512 GB RAM per node, Dual AMD Processors (128 CPU cores per node). | Weather modelling, climate simulations, and large-scale parallel scientific computing. |
| **GPU Cluster**<br>(12 Nodes) | Four NVIDIA A100 GPUs per GPU node. | Deep Learning, CUDA-accelerated ML, and GPU-intensive simulations. |

## 9. Best Practices

- **Resource Awareness:** Always specify exact walltime and node requirements in PBS scripts to minimize queue wait time.
- **Storage Management:** Periodically clear temporary data from the Scratch directory.
- **Environment Integrity:** Never install packages globally; use isolated virtual environments.
- **Job Monitoring:** Use `qstat` to monitor job status and verify hardware utilization.
- **Pre-Execution:** Inspect PBS `.o` and `.e` files immediately after completion to debug execution errors.

## 10. Summary

CHAMP is a centralized High Performance Computing facility designed to support advanced scientific computing through parallel processing. By combining secure access, centralized scheduling, dedicated CPU/GPU resources, and flexible software environments, it enables researchers to execute workloads impractical for conventional systems. During this internship, mastery of the CHAMP ecosystem—from secure entry and environment configuration to the management of the PBS job lifecycle—was foundational to successfully deploying machine learning and environmental simulation models.
