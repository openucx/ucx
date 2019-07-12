# Introduction

This project uses Azure Pipelines a GitHub check to validate pull requests
prior to merging. Each time a pull request is updated AZP will spawn VMs and
run compiles and tests based on the instructions in the
buildlib/azure-pipelines.yml file.

The test console output is linked from the GitHub check integration.

Azure Pipelines is linked to the UCF Consortium's Azure Tenant:

   https://portal.azure.com

And runs inside the Azure Dev Ops Organization:

   https://dev.azure.com/ucfconsort

As the UCX project:

   https://dev.azure.com/ucfconsort/ucx
