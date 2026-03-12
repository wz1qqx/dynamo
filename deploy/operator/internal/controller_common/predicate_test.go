/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller_common

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func TestGetDiscoveryBackend(t *testing.T) {
	tests := []struct {
		name        string
		configValue configv1alpha1.DiscoveryBackend
		annotations map[string]string
		want        configv1alpha1.DiscoveryBackend
	}{
		{
			name:        "defaults to kubernetes when config and annotation are empty",
			configValue: "",
			annotations: nil,
			want:        configv1alpha1.DiscoveryBackendKubernetes,
		},
		{
			name:        "uses config when annotation is absent",
			configValue: configv1alpha1.DiscoveryBackendEtcd,
			annotations: nil,
			want:        configv1alpha1.DiscoveryBackendEtcd,
		},
		{
			name:        "uses annotation override when set",
			configValue: configv1alpha1.DiscoveryBackendKubernetes,
			annotations: map[string]string{commonconsts.KubeAnnotationDynamoDiscoveryBackend: "etcd"},
			want:        configv1alpha1.DiscoveryBackendEtcd,
		},
		{
			name:        "ignores empty annotation and falls back to default",
			configValue: "",
			annotations: map[string]string{commonconsts.KubeAnnotationDynamoDiscoveryBackend: ""},
			want:        configv1alpha1.DiscoveryBackendKubernetes,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetDiscoveryBackend(tt.configValue, tt.annotations)
			if got != tt.want {
				t.Fatalf("GetDiscoveryBackend() = %q, want %q", got, tt.want)
			}
		})
	}
}
