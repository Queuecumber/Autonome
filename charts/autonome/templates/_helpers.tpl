{{- define "autonome.fullname" -}}
{{- printf "%s-%s" .Release.Name .name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "autonome.labels" -}}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "autonome.image" -}}
{{ .root.Values.image.registry }}/{{ .svc.image.name }}:{{ default .root.Values.image.tag .svc.image.tag }}
{{- end -}}

{{- define "autonome.resources" -}}
{{- $svcResources := default dict .svc.resources -}}
{{- $merged := mergeOverwrite (deepCopy .root.Values.resources) $svcResources -}}
{{- toYaml $merged -}}
{{- end -}}

{{- define "autonome.secretName" -}}
{{- if and (not .Values.secrets.create) .Values.secrets.existingName -}}
{{ .Values.secrets.existingName }}
{{- else -}}
{{ .Release.Name }}-secrets
{{- end -}}
{{- end -}}
