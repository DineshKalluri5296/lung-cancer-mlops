Identity Providers
NAme:token.actions.githubusercontent.com
Audience  sts.amazonaws.com


{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::821656895219:oidc-provider/token.actions.githubusercontent.com"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                    "token.actions.githubusercontent.com:sub": "repo:DineshKalluri5296@237681550/lung-cancer-mlops@1342475142:ref:refs/heads/main"
                }
            }
        }
    ]


====================KMS============================
  
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "UseLungCancerKMSKey",
            "Effect": "Allow",
            "Action": [
                "kms:Decrypt",
                "kms:Encrypt",
                "kms:GenerateDataKey",
                "kms:DescribeKey"
            ],
            "Resource": "arn:aws:kms:ap-south-1:821656895219:key/9e7a6381-5b97-4901-86d2-ed729ffa0bcf"
        }
    ]
}

  ===================s3 LungCancerMLOpsS3Policy ============================
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ListProjectBucket",
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": "arn:aws:s3:::lung-cancer-mlops-dinesh-202612"
        },
        {
            "Sid": "ReadDataset",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::lung-cancer-mlops-dinesh-202612/data/*"
        },
        {
            "Sid": "WriteArtifacts",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::lung-cancer-mlops-dinesh-202612/artifacts/*"
        },
        {
            "Sid": "ReadWriteModels",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::lung-cancer-mlops-dinesh-202612/models/*"
        }

  
