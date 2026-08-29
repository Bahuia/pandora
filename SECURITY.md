# Security policy

Please report suspected vulnerabilities privately through GitHub's security
advisory feature for this repository. Do not include credentials, private data,
or working exploit details in a public issue.

The generated-code executor applies AST checks and subprocess resource limits,
but it is a research sandbox rather than a hardened isolation boundary. Run
untrusted model output only inside a container or virtual machine with minimal
filesystem and network access.

Only the latest released version receives security fixes. API credentials must
be supplied through provider environment variables and must never be logged or
stored in result files.
