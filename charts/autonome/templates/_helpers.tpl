{{/*
Resource name = <release>-<service>. Keeping it release-prefixed so two
agents can share a namespace without colliding (though typically they
won't).
*/}}
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

{{/*
Resolves the name of the Secret holding OPENAI_API_KEY / MATRIX_PASSWORD /
SEARCH_API_KEY. When the chart owns the Secret it's <release>-secrets;
otherwise the user-specified existing Secret (falling back to the same
default name).
*/}}
{{- define "autonome.secretName" -}}
{{- if and (not .Values.secrets.create) .Values.secrets.existingName -}}
{{ .Values.secrets.existingName }}
{{- else -}}
{{ .Release.Name }}-secrets
{{- end -}}
{{- end -}}
