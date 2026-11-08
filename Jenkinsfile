// broker_parrot CI: every pushed branch runs scripts/ci/test.sh — the whole pytest suite on SQLite and on a throwaway
// Postgres container — and reports ONE GitHub commit status, context `jenkins/ci`, whose link is this
// build. A PR shows its head branch's status.
//
// Jenkins side: a multibranch job reading this file from SCM, an executor with a Docker CLI on a daemon it may start
// containers on, and the Secret Text credential `github-pat` (a token allowed to post statuses on this repo).
// The repo the status goes to is read from the checkout's origin remote — nothing about the deployment is written here.

pipeline {
  agent any

  options {
    timestamps()
    disableConcurrentBuilds()
    timeout(time: 30, unit: 'MINUTES')
    buildDiscarder(logRotator(numToKeepStr: '30'))
  }

  triggers {
    githubPush()
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
        script {
          env.GIT_SHA = sh(script: 'git rev-parse HEAD', returnStdout: true).trim()
          def remoteRepo = sh(returnStdout: true, script: '''
            url=$(git config --get remote.origin.url)
            repo=$(printf '%s' "$url" | sed -nE 's#^(https://|git@|ssh://git@)github\\.com[:/]([^/]+/[^/]+)$#\\2#p' | sed 's#\\.git$##')
            [ -n "$repo" ] || { echo "origin $url is not a GitHub repo — nowhere to post the status" >&2; exit 1; }
            printf '%s' "$repo"
          ''').trim()
          // The remote may name the repo by a pre-rename name: git follows GitHub's redirect, but a status POST to the
          // old name gets HTTP 307. Ask GitHub for the repo's CURRENT full_name (GET follows the redirect) and post there.
          withCredentials([string(credentialsId: 'github-pat', variable: 'GH_TOKEN')]) {
            env.GH_REPO = sh(returnStdout: true, script: """
              set +x
              name=\$(curl -sSfL -H "Authorization: Bearer \$GH_TOKEN" -H "Accept: application/vnd.github+json" \
                "https://api.github.com/repos/${remoteRepo}" | sed -nE 's#^  \"full_name\": \"([^\"]+)\",\$#\\1#p')
              [ -n "\$name" ] || { echo "GitHub did not resolve repo ${remoteRepo} — check credential github-pat" >&2; exit 1; }
              printf '%s' "\$name"
            """).trim()
          }
          echo "Building ${env.GH_REPO}@${env.GIT_SHA}"
        }
        notifyGitHub('pending', 'SQLite + Postgres suites running')
      }
    }

    // the suite image (deps cached by a pyproject hash) once, so the two parallel gates reuse it instead of racing to build it
    stage('Suite image') { steps { sh 'scripts/ci/test.sh image' } }

    stage('Gates') {
      parallel {
        stage('SQLite suite')   { steps { sh 'scripts/ci/test.sh sqlite' } }
        stage('Postgres suite') { steps { sh 'scripts/ci/test.sh pg' } }
      }
    }
  }

  post {
    success  { notifyGitHub('success', 'SQLite + Postgres suites passed') }
    failure  { notifyGitHub('failure', 'CI failed — see the build log') }
    unstable { notifyGitHub('failure', 'CI unstable') }
    aborted  { notifyGitHub('error', 'Build aborted') }
    cleanup  { cleanWs() }
  }
}

// POST /repos/:repo/statuses/:sha. A status that cannot be posted FAILS the build: a green build GitHub never heard
// about would look exactly like a PR with no CI.
void notifyGitHub(String state, String description) {
  if (!env.GIT_SHA || !env.GH_REPO) { error("no commit/repo known yet — cannot post GitHub status ${state}") }
  withCredentials([string(credentialsId: 'github-pat', variable: 'GH_TOKEN')]) {
    writeFile file: '.github-status.json', text: groovy.json.JsonOutput.toJson([
      state: state, target_url: env.BUILD_URL, description: description, context: 'jenkins/ci'
    ])
    def code = sh(returnStdout: true, script: '''
      set +x
      curl -sS -o /dev/null -w '%{http_code}' -X POST \
        -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" --data @.github-status.json \
        "https://api.github.com/repos/$GH_REPO/statuses/$GIT_SHA"
      rm -f .github-status.json
    ''').trim()
    if (code != '201') { error("GitHub refused status ${state} for ${env.GH_REPO}@${env.GIT_SHA} (HTTP ${code}) — check credential github-pat") }
    echo "GitHub status ${state} posted for ${env.GIT_SHA.take(8)}"
  }
}
