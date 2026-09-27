// SPDX-License-Identifier: GPL-2.0-only
/*
 * Vendor hooks of arm-smmu-impl.c that the ZynqMP MMU-500 ("arm,mmu-500")
 * never takes: the NVIDIA one only runs for Tegra compatibles, the Qualcomm
 * one only with CONFIG_ARM_SMMU_QCOM.  The upstream objects that define them
 * (arm-smmu-nvidia.c, arm-smmu-qcom.c) are not built here.
 */
#include "arm-smmu.h"

struct arm_smmu_device *nvidia_smmu_impl_init(struct arm_smmu_device *smmu)
{
	return smmu;
}

struct arm_smmu_device *qcom_smmu_impl_init(struct arm_smmu_device *smmu)
{
	return smmu;
}
