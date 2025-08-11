---
title: Guide title
shortTitle: <subject> # Max 31 characters
intro: 'Article intro. See tips for a great intro below'
product: "{{ optional product callout }}"
type: tutorial
topics:
  - <topic> # One or more from list of allowed topics: https://github.com/github/docs/blob/main/data/allowed-topics.js
versions:
  - <version>
---

{% comment %}
- Great intros clarify who the guide is intended for, state what the user will accomplish, and state the technologies that will be used.
- Intros are typically 1-3 sentence summaries, with a longer "Introduction" section that follows.
- Remove these comments from your article file when you're done writing
{% endcomment %}

## Introduction

{% comment %}
The language guide introduction should include the following in a short paragraph -
- Clarify audience.
- State prerequisites and prior knowledge needed.
- Should the user have read any other articles?
- State what the user will accomplish or build and the user problem it solves.
{% endcomment %}

## Starting with the <language> workflow template

{% comment %}
Language guides typically walk through and build upon a workflow template. If that format doesn't work, you can include a boilerplate workflow.
- Link to the GitHub Actions CI workflow template as the boilerplate reference code and then walk through and build on that code in this guide - https://github.com/actions/starter-workflows/tree/main/ci
- Provide instructions for adding the workflow template to a repository.
- Include the starter template workflow code.
{% endcomment %}

## Running on different operating systems

{% comment %}
Include a brief overview of how to choose the runner environment. These should be alternatives to what operating system is presented in the workflow template/boilerplate template.
{% endcomment %}

## Configuring the <language> version

{% comment %}
- Describe when and how to use available setup actions that configure the version of the language on the runner (ex. actions/setup-node).
- How does the setup action configure the version and what happens when the version isn't supported in the environment. What is the default version, when no version is configured.
- Include any additional features the setup action might provide that are useful to CI.
- If applicable, provide examples of configuring exact versions or major/minor versions.
- Include information about software already installed on GitHub-hosted runners or software configuration necessary to build and test the project.
- Provide examples of configuring matrix strategies.
- Link out to any docs about available software on the GitHub-hosted runners. (Ex. https://docs.github.com/en/actions/reference/software-installed-on-github-hosted-runners).
- Include code samples.
{% endcomment %}

## Installing dependencies

{% comment %}
- Include example of installing dependencies to prepare for building and testing.
- Are there any dependencies or scenarios where people might need to install packages globally?
- Include examples of common package managers.
- If the language is supported by GitHub Packages, include an example installing dependencies from GitHub.
- Include code samples.
{% endcomment %}

## Caching dependencies

{% comment %}
Include an example of restoring cached dependencies. We'll want to link out to the article about caching for more information (https://docs.github.com/en/actions/configuring-and-managing-workflows/caching-dependencies-to-speed-up-workflows).
{% endcomment %}

## Building your code

{% comment %}
- Include any compile steps.
- Include any test commands.
- Note that you can use the same commands that your repository needs to build and test your code by simply replacing the commands in the `run` keyword.
- Include any basic examples or commands specific to test frameworks.
- Include any common databases or services that might be needed. If so, we can link out to the services guides in the docs (https://docs.github.com/en/actions/configuring-and-managing-workflows/using-databases-and-service-containers).
{% endcomment %}

## Packaging workflow data as artifacts

{% comment %}
This section can simply link out to https://docs.github.com/en/actions/configuring-and-managing-workflows/persisting-workflow-data-using-artifacts or provide additional information about which artifacts might be typical to upload for a CI workflow.
{% endcomment %}
