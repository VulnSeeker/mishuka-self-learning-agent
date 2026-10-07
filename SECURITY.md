# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in Onyx, please report it privately.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please use GitHub's private vulnerability reporting:

1. Go to the **Security** tab of this repository
2. Click **"Report a vulnerability"**
3. Fill in the details of the issue

Alternatively, you can email the maintainer directly at
`vulnseeker@users.noreply.github.com`.

## What to Include

Please provide:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Affected version(s)
- Any suggested fix or mitigation

## Response Timeline

- **Acknowledgement**: within 48 hours
- **Initial assessment**: within 5 business days
- **Fix or mitigation**: best effort, based on severity

## Scope

In-scope:
- Arbitrary code execution in the sandbox
- Authentication or authorization bypass in the API
- Prompt injection leading to data exfiltration
- Path traversal or unauthorized file access
- Leaked secrets or credentials in logs
- Dependency vulnerabilities in runtime requirements

Out-of-scope:
- Vulnerabilities in third-party LLM providers
- Issues in development-only dependencies
- Social engineering attacks
- Denial-of-service via legitimate high-volume API usage

## Security Best Practices for Users

When deploying Onyx:

- **Never commit your `.env` file** or API keys to version control
- Set `ONYX_API_TOKEN` in production to require authentication
- Run the container as a **non-root user** (the provided Dockerfile does this)
- Enable the **Docker sandbox** (`USE_DOCKER_SANDBOX=true`) for untrusted code
- Rotate API keys regularly
- Review the `ALLOWED_TARGET_DOMAINS` list before running OSINT tasks
- Keep dependencies up to date

## Disclosure Policy

We follow coordinated disclosure. Once a fix is available, we will:

1. Publish a security advisory on GitHub
2. Release a patched version
3. Credit the reporter (unless they prefer to remain anonymous)

Thank you for helping keep Onyx and its users safe.
