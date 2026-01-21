# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email the maintainers directly with details
3. Include steps to reproduce if possible
4. Allow reasonable time for a fix before public disclosure

## What to Report

- Authentication or authorization flaws
- Data exposure risks
- Injection vulnerabilities
- Dependency vulnerabilities with known exploits

## Response Timeline

- **Acknowledgment:** Within 48 hours
- **Initial assessment:** Within 1 week
- **Fix timeline:** Depends on severity, typically 30-90 days

## Security Best Practices for Users

- Never commit API keys or secrets to version control
- Use environment variables for sensitive configuration
- Keep dependencies updated
- Review the `.env.example` file for required secrets
