# microplex-us docs

- [Architecture](./architecture.md)
- [Canonical pipeline stages](./pipeline-stages.md)
- [Stage contracts and manifests](./stage-contracts.md)
- [API reference](./api.md)
- [Source semantics](./source-semantics.md)
- [Imputation conditioning contract](./imputation-conditioning-contract.md)
- [Benchmarking](./benchmarking.md)
- [Methodology ledger](./methodology-ledger.md)
- [PolicyEngine oracle compatibility path](./policyengine-oracle-compatibility.md)
- [PE construction parity](./pe-construction-parity.md)
- [Superseding `policyengine-us-data`](./superseding-policyengine-us-data.md)
- [Hugging Face artifact publishing](./huggingface-artifact-publishing.md)

This doc set is intentionally technical. It is meant to answer seven questions:

1. What is the current architecture?
2. How do source semantics and variable semantics drive donor integration?
3. What is structurally required in imputation conditioning, and what is still
   experimental?
4. Which construction contracts currently match PE, and which are only
   compatible?
5. How do we measure progress against `policyengine-us-data` on real targets?
6. What is the actual roadmap for fully superseding `policyengine-us-data`?
7. Which methodological choices are currently canonical, provisional, or open?

The docs describe the code that exists today. They do not try to freeze a final
paper narrative while the architecture is still moving.
